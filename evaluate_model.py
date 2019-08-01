from load_data import *
from PIXOR import PIXOR
from shapely.geometry import Polygon
from config import *
from scipy.linalg import block_diag


#######################
# reshape predictions #
#######################

def process_regression_target(valid_reg_predictions, validity_mask):
    """
        Convert the raw regression prediction into camera coordinates of a bounding box.
        :param valid_reg_predictions: raw regression output | shape: [N_VALID, OUTPUT_DIM_REG]
        :param validity_mask: mask of predictions indices with confidence score higher than threshold. Required for the
                              calculation of the camera coordinates of the predicted bounding box | shape: [INPUT_DIM_0, INPUT_DIM_1]
        :return: point_cloud_box_predictions: predicted box coordinates | shape: [N_VALID, 8]
        """

    # get camera coordinates of the pixel corner for all valid predictions
    x_lin = np.linspace(VOX_Y_MIN, VOX_Y_MAX - 0.4, OUTPUT_DIM_1)
    y_lin = np.linspace(VOX_X_MAX, VOX_X_MIN + 0.4, OUTPUT_DIM_0)
    px_x, px_y = np.meshgrid(x_lin, y_lin)
    px_x = px_x.reshape((-1,))[validity_mask]
    px_y = px_y.reshape((-1,))[validity_mask]

    # get the camera coordinates of the center of the predicted bounding boxes
    center_x = np.broadcast_to((px_x - valid_reg_predictions[:, 2]).reshape(-1, 1), (valid_reg_predictions.shape[0], 4))
    center_y = np.broadcast_to((px_y - valid_reg_predictions[:, 3]).reshape(-1, 1), (valid_reg_predictions.shape[0], 4))
    centers = np.hstack((center_x, center_y))

    # get the predicted angles for all bounding boxes
    prediction_angles = np.arctan2(valid_reg_predictions[:, 1], valid_reg_predictions[:, 0])
    prediction_cos = np.cos(prediction_angles)
    prediction_sin = np.sin(prediction_angles)

    # build a block diagonal rotation matrix to rotate all predicted bounding boxes
    rot_matrices = np.stack((prediction_cos, prediction_sin, -prediction_sin, prediction_cos), axis=1).reshape((-1, 2))
    rot_matrices = [rot_matrices[i:i + 2, :] for i in range(0, rot_matrices.shape[0], 2)]
    block_rot_matrix = block_diag(*rot_matrices)

    # get predicted width and length for all bounding boxes
    width = np.exp(valid_reg_predictions[:, 4])
    length = np.exp(valid_reg_predictions[:, 5])

    # get all bounding box corners
    x_corners = np.stack((length / 2, length / 2, -length / 2, -length / 2), axis=1)
    y_corners = np.stack((-width / 2, width / 2, width / 2, -width / 2), axis=1)
    corners = np.stack((x_corners, y_corners), axis=1).reshape((-1, 4))

    # rotate the bounding boxes
    corners = np.dot(block_rot_matrix, corners).reshape(-1, 8)
    # translate the bounding boxes
    point_cloud_box_predictions = corners + centers

    return point_cloud_box_predictions


##########################
# non-maximum suppression #
##########################

def perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):
    """
        Perform Non-Maximum Suppression to eliminate overlapping predictions.
        :param valid_class_predictions: all confidence scores higher than the threshold | shape: [N_VALID, ]
        :param valid_box_predictions: all corresponding bounding box predictions | shape: [N_VALID, 8]
        :param nms_threshold: threshold for maximum overlap between two bounding boxes
        :return: sorted_class_predictions: remaining confidence scores | shape: [N_FINAL, ]
                 sorted_box_predictions: remaining box predictions | shape: [N_FINAL, 8]
        """

    # sort the detections such that the entry with the maximum confidence score is at the top
    sorted_indices = np.argsort(valid_class_predictions)[::-1]
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]

    for i in range(sorted_box_predictions.shape[0]):
        # get the IOUs of all boxes with the currently most certain bounding box
        try:
            ious = np.zeros((sorted_box_predictions.shape[0]))
            ious[i + 1:] = bbox_iou(sorted_box_predictions[i, :], sorted_box_predictions[i + 1:, :])
        except ValueError:
            break
        except IndexError:
            break

        # eliminate all detections which have IoU > threshold
        overlap_mask = np.where(ious < nms_threshold, True, False)
        sorted_box_predictions = sorted_box_predictions[overlap_mask]
        sorted_class_predictions = sorted_class_predictions[overlap_mask]

    return sorted_class_predictions, sorted_box_predictions


####################
# bounding box IoU #
####################

def bbox_iou(box1, boxes):
    """
    Compute the bounding box IoUs between the given bounding box "box1" and a number of bounding boxes in "boxes"
    :param box1: given bounding box | shape: [8, ]
    :param boxes: bounding boxes for which IoU with box1 is to be computed | shape: [N, 8]
    :return: IoUs for each box in "boxes" with "box1" | shape: [N, ]
    """

    # currently inspected box
    box1 = box1.reshape((2, 4)).T
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((2, 4)).T
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou = inter_area / (area_1 + area_2 - inter_area)
        ious[box_id] = iou

    return ious


#######################
# process predictions #
#######################

def process_predictions(batch_predictions, confidence_threshold=0.2, nms_threshold=0.05):
    """
    Process the raw network output into thresholded and non-overlapping final bounding box predictions for all samples in
    the batch.
    :param batch_predictions: raw network output for entire batch |
           shape: [batch_size, OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_CLA+OUTPUT_DIM_REG]
    :param confidence_threshold: minimum confidence score in order for a prediction to be considered valid
    :param nms_threshold: threshold for maximum IoU in order for two boxes to be considered non-overlapping
    :return: final_batch_predictions: processed bounding box predictions | shape: [N_FINAL, 10]
    """

    # inverse normalization
    batch_predictions[:, :, :, :-1] = (batch_predictions[:, :, :, :-1] * REG_STD) + REG_MEAN

    # process targets and perform NMS for each prediction in batch
    final_batch_predictions = None  # store final bounding box predictions
    for point_cloud_id in range(batch_predictions.shape[0]):

        # get all predictions for single point cloud
        point_cloud_predictions = batch_predictions[point_cloud_id]

        # reshape predictions
        point_cloud_predictions = point_cloud_predictions.reshape(
            (OUTPUT_DIM_0 * OUTPUT_DIM_1, OUTPUT_DIM_CLA + OUTPUT_DIM_REG))

        # separate classification and regression output
        point_cloud_class_predictions = point_cloud_predictions[:, -1]
        point_cloud_reg_predictions = point_cloud_predictions[:, :-1]

        # get valid detections
        validity_mask = np.where(point_cloud_class_predictions > confidence_threshold, True, False)
        valid_reg_predictions = point_cloud_reg_predictions[validity_mask]
        valid_class_predictions = point_cloud_class_predictions[validity_mask]

        # continue if no valid predictions
        if valid_reg_predictions.shape[0]:
            valid_box_predictions = process_regression_target(valid_reg_predictions, validity_mask)
        else:
            continue

        # perform Non-Maximum Suppression
        final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,
                                                                     nms_threshold)

        # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
        point_cloud_ids = np.ones((final_box_predictions.shape[0], 1)) * point_cloud_id
        final_point_cloud_predictions = np.hstack((point_cloud_ids, final_class_predictions[:, np.newaxis],
                                                   final_box_predictions))

        # stack final predictions for all point clouds in batch
        if final_batch_predictions is None:
            final_batch_predictions = final_point_cloud_predictions
        else:
            final_batch_predictions = np.vstack((final_batch_predictions, final_point_cloud_predictions))

    return final_batch_predictions


##################
# evaluate model #
##################


def evaluate_model(model, data_loader, distance_ranges, iou_thresholds):
    """
    Evaluate the performance of a trained model on the test set. Store an "eval_dict" with all relevant performance
    metrics for further inspection and plotting of graphs.
    :param model: trained PIXOR model
    :param data_loader: PyTorch data loader containing the test dataset
    :param distance_ranges: list of all distance ranges for which evaluation should be performed in descending order.
           Example:  [50] -> only detections within 0-50m, [30, 50] -> separate evaluation of 0-30m and 0-50m detections
    :param iou_thresholds: list of IoU thresholds for which the model should be evaluated.
           Example: [0.5] -> detections with an IoU of 0.5 or higher with a ground truth box are regarded as positive
    :return: eval_dict: dictionary containing precision, recall and AP for each threshold at each distance + mAP
             averaged over all thresholds for each distance range
    """

    model.eval()  # set model to evaluate mode

    # set up evaluation dictionary
    eval_dict = {distance_range: {'targets': {threshold: [] for threshold in iou_thresholds}, 'scores': [], 'n_labels': 0}
                 for distance_range in distance_ranges}

    # iterate over all batches in test dataset
    for batch_id, (batch_data, batch_labels, batch_calib) in enumerate(data_loader):

        with torch.set_grad_enabled(False):

            # forward pass
            batch_predictions = model(batch_data)

            # convert network output to numpy for further processing
            batch_predictions = np.transpose(batch_predictions.detach().numpy(), (0, 2, 3, 1))

            # get final bounding box predictions
            final_box_predictions = process_predictions(batch_predictions)

            # iterate over all point clouds in batch
            for point_cloud_id in range(batch_data.size(0)):

                # in case of valid predictions, get predictions for currently inspected point cloud
                if final_box_predictions is not None:
                    point_cloud_predictions = np.vstack(
                        [predictions for predictions in final_box_predictions if predictions[0] == point_cloud_id])
                else:
                    point_cloud_predictions = None

                # get coordinates of all relevant ground truth boxes
                ground_truth_box_corners = None
                for label in batch_labels[point_cloud_id]:
                    # only consider annotations for class "Car"
                    if label.type == 'Car':
                        # compute corners of the bounding box
                        _, bbox_corners_camera_coord = kitti_utils.compute_box_3d(label, batch_calib[id].P)
                        bbox_corners_camera_coord = np.hstack((bbox_corners_camera_coord[:4, 0], bbox_corners_camera_coord[:4, 2]))
                        if ground_truth_box_corners is None:
                            ground_truth_box_corners = bbox_corners_camera_coord
                        else:
                            ground_truth_box_corners = np.vstack((ground_truth_box_corners, bbox_corners_camera_coord))

                assert np.all(np.diff(distance_ranges) <= 0)  # check that distance ranges monotonically decrease
                # iterate over all distance ranges
                for distance_range in distance_ranges:

                    # valid predictions and labels exist for the currently inspected point cloud
                    if ground_truth_box_corners is not None and point_cloud_predictions is not None:

                        # remove all predictions and labels outside of the specified range
                        max_distance_predictions = np.max(point_cloud_predictions[:, 6:], axis=1)
                        max_distance_ground_truth = np.max(ground_truth_box_corners[:, 4:], axis=1)
                        distance_mask_predictions = np.where(max_distance_predictions <= distance_range, True, False)
                        distance_mask_ground_truth = np.where(max_distance_ground_truth <= distance_range, True, False)
                        point_cloud_predictions = point_cloud_predictions[distance_mask_predictions]
                        ground_truth_box_corners = ground_truth_box_corners[distance_mask_ground_truth]

                        # valid predictions and labels exist inside the specified range
                        if point_cloud_predictions.shape[0] and ground_truth_box_corners.shape[0]:

                            # compute IoUs of all predictions with all bounding boxes and store the corresponding
                            # confidence scores
                            ious = np.zeros((point_cloud_predictions.shape[0], ground_truth_box_corners.shape[0]))
                            confidence_scores = np.zeros((point_cloud_predictions.shape[0], ground_truth_box_corners.shape[0]))
                            for pid, prediction in enumerate(point_cloud_predictions):
                                ious[pid, :] = bbox_iou(prediction[2:], ground_truth_box_corners)
                                confidence_scores[pid, np.argmax(ious[pid, :])] = prediction[1]

                            # iterate over all thresholds to compute the number of positive detections
                            for threshold in iou_thresholds:
                                eval_dict[distance_range]['targets'][threshold].extend(np.where(np.max(ious, axis=1) > threshold, 1, 0))

                            # store the predictions' confidence scores and the number of labels
                            eval_dict[distance_range]['scores'].extend(np.max(confidence_scores, axis=1))
                            eval_dict[distance_range]['n_labels'] += ground_truth_box_corners.shape[0]

                        # valid predictions exist but no labels
                        elif point_cloud_predictions.shape[0] and not ground_truth_box_corners.shape[0]:

                            # store all predictions as negative detections
                            for threshold in iou_thresholds:
                                eval_dict[distance_range]['targets'][threshold].extend(np.zeros((point_cloud_predictions.shape[0],)))

                            # store the predictions' confidence scores
                            eval_dict[distance_range]['scores'].extend(point_cloud_predictions[:, 1])

                        # valid labels exist but no predictions
                        elif not point_cloud_predictions.shape[0] and ground_truth_box_corners.shape[0]:

                            # increase the number of labels
                            eval_dict[distance_range]['n_labels'] += ground_truth_box_corners.shape[0]

                        # neither labels nor predictions exist
                        else:
                            continue

                    # valid predictions exist for the currently inspected point cloud but no labels
                    elif ground_truth_box_corners is None and point_cloud_predictions is not None:

                        # remove all predictions outside of the specified range
                        max_distance_predictions = np.max(point_cloud_predictions[:, 6:], axis=1)
                        distance_mask_predictions = np.where(max_distance_predictions <= distance_range, True, False)
                        point_cloud_predictions = point_cloud_predictions[distance_mask_predictions]

                        # valid predictions exist inside the specified range
                        if point_cloud_predictions.shape[0]:

                            # store all predictions as negative detections
                            for threshold in iou_thresholds:
                                eval_dict[distance_range]['targets'][threshold].extend(np.zeros((point_cloud_predictions.shape[0],)))
                            # store the predictions' confidence scores
                            eval_dict[distance_range]['scores'].extend(point_cloud_predictions[:, 1])

                    # valid labels exist for the currently inspected point cloud but no predictions
                    elif point_cloud_predictions is None and ground_truth_box_corners is not None:

                        # remove all labels outside of the specified range
                        max_distance_ground_truth = np.max(ground_truth_box_corners[:, 6:], axis=1)
                        distance_mask_ground_truth = np.where(max_distance_ground_truth <= distance_range, True, False)
                        ground_truth_box_corners = ground_truth_box_corners[distance_mask_ground_truth]

                        # valid labels exist inside specified range
                        if ground_truth_box_corners.shape[0]:
                            # increase the number of labels
                            eval_dict[distance_range]['n_labels'] += ground_truth_box_corners.shape[0]
                        else:
                            continue

                    # neither valid predictions nor labels exist for the currently inspected point cloud
                    else:
                        continue

                print('++++++++++++++++++++++++++++++')
                print('Analyze Point Cloud {:d}/{:d}'.format(batch_id*batch_size+point_cloud_id+1, data_loader.dataset.__len__()))

    # compute performance metrics for each evaluated distance range
    for distance_range in distance_ranges:

        # sort the predictions according to the confidence score
        sorted_indices = np.argsort(eval_dict[distance_range]['scores'])[::-1]
        eval_dict[distance_range]['scores'] = np.array(eval_dict[distance_range]['scores'])[sorted_indices]

        # iterate over all thresholds
        for threshold in iou_thresholds:

            # sort the targets to match the confidence scores
            eval_dict[distance_range]['targets'][threshold] = np.array(eval_dict[distance_range]['targets'][threshold])[sorted_indices]

            # create a dict to store all relevant performance metrics
            performance_dict = {}

            # compute recall
            recall = list(np.cumsum(eval_dict[distance_range]['targets'][threshold]) / eval_dict[distance_range]['n_labels'])
            recall.insert(0, 0.)  # start with 0
            recall = np.array(recall)  # convert to numpy array for further processing
            performance_dict['recall'] = recall

            # compute precision
            precision = [np.sum(eval_dict[distance_range]['targets'][threshold][:i + 1]) / (i + 1) for i in range(len(eval_dict[distance_range]['targets'][threshold]))]
            precision.insert(0, 0.)  # start with 0
            precision = np.array(precision)
            performance_dict['precision'] = precision

            # compute average precision
            indices = np.where(recall[:-1] != recall[1:])[0] + 1
            average_precision = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
            performance_dict['AP'] = average_precision

            # add performance_dict for currently inspected threshold to eval_dict
            eval_dict[distance_range][threshold] = performance_dict

        # compute mAP as the average of the AP over all IoU-thresholds
        average_precisions = [eval_dict[distance_range][key]['AP'] for key in eval_dict[distance_range] if not isinstance(key, str)]
        eval_dict[distance_range]['mAP'] = sum(average_precisions) / len(average_precisions)

    return eval_dict


if __name__ == '__main__':

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # evaluation parameters
    batch_size = 1

    # create data loader
    root_dir = 'Data/'
    data_loader = load_dataset(root=root_dir, batch_size=batch_size, device=device, test_set=True)

    # create model
    pixor = PIXOR()
    n_epochs_trained = 17
    pixor.load_state_dict(torch.load('Models/PIXOR_Epoch_' + str(n_epochs_trained) + '.pt', map_location=device))

    # evaluate model
    eval_dict = evaluate_model(pixor, data_loader, distance_ranges=[70, 50, 30], iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9])

    # add identifier to dictionary
    eval_dict['epoch'] = n_epochs_trained

    # save evaluation dictionary
    np.savez('eval_dict_epoch_' + str(n_epochs_trained) + '.npz', eval_dict=eval_dict)
