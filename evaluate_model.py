from load_data import *
from PIXOR_Net import PIXOR
import torch.nn as nn
import cv2
import math
from shapely.geometry import Polygon
from config import *


############
# bbox IoU #
############

def bbox_iou(box1, boxes):

    # currently inspected box
    box1 = box1.view([2, 4]).permute([1, 0])  # reshape bounding box corners
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]), (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = torch.zeros([boxes.size(0)])
    for box_id in range(boxes.size(0)):
        box2 = boxes[box_id, :]
        box2 = box2.view([2, 4]).permute([1, 0])  # reshape bounding box corners
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]), (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou_bev = inter_area / (area_1 + area_2 - inter_area)
        ious[box_id] = iou_bev

    return ious


#######################
# process predictions #
#######################


def process_predictions(predictions, confidence=0.6, nms_conf=0.3):
    # print('Prediction Size: ', predictions.size())
    # pass class prediction through a sigmoid activation
    sigmoid = nn.Sigmoid()
    classification_prediction = sigmoid(predictions[:, :, :, -1])
    predictions[:, :, :, -1] = classification_prediction
    print('Min Confidence Score: {:.4f}'.format(torch.min(classification_prediction).item()))
    print('Max Confidence Score: {:.4f}'.format(torch.max(classification_prediction).item()))

    # create a mask for all predictions with confidence higher than threshold
    conf_mask = (classification_prediction > confidence).float().unsqueeze(3)
    # zero out all predictions with confidence lower than threshold
    predictions = predictions * conf_mask

    print('Number of valid detections:', int(torch.sum(conf_mask).item()))
    print('Number of invalid detections: ', int(conf_mask.numel() - torch.sum(conf_mask).item()))

    # inverse normalization
    predictions[:, :, :, :-1] = (predictions[:, :, :, :-1] * torch.tensor(REG_STD).float()) + torch.tensor(REG_MEAN).float()
    prediction_mask = (predictions[:, :, :, :-1] != torch.tensor(REG_MEAN).float()).float()
    predictions[:, :, :, :-1] = predictions[:, :, :, :-1] * prediction_mask

    # perform NMS for each prediction in batch
    batch_size = predictions.size(0)
    # store final predictions
    final_predictions = None
    for idp in range(batch_size):
        # get all predictions for single point cloud
        prediction = predictions[idp]

        # store final predictions after NMS
        final_prediction = None
        for i in range(prediction.size(0)):
            for j in range(prediction.size(1)):
                regression_prediction = prediction[i, j, :-1]
                classification_prediction = prediction[i, j, -1].unsqueeze(0).unsqueeze(1)

                if torch.sum(torch.abs(regression_prediction)):

                    # get angle from prediction
                    angle = torch.tensor(math.atan2(regression_prediction[1], regression_prediction[0]))
                    c = torch.cos(angle)
                    s = torch.sin(angle)
                    R = torch.tensor([[c, s], [-s, c]])

                    # get location of bounding box
                    pixel_location_x_m = (j * (INPUT_DIM_1 / OUTPUT_DIM_1 * VOX_Y_DIVISION)) + VOX_Y_MIN
                    pixel_location_y_m = VOX_X_MAX - (i * (INPUT_DIM_0 / OUTPUT_DIM_0 * VOX_X_DIVISION))
                    center_x = pixel_location_x_m - regression_prediction[2]
                    center_y = pixel_location_y_m - regression_prediction[3]

                    # get size of bounding box
                    width = torch.exp(regression_prediction[4])
                    length = torch.exp(regression_prediction[5])

                    # get corners of bounding box
                    x_corners = torch.tensor([length / 2, length / 2, -length / 2, -length / 2]).unsqueeze(0)
                    y_corners = torch.tensor([-width / 2, width / 2, width / 2, -width / 2]).unsqueeze(0)

                    # rotate and translate bounding box
                    corners_bev = torch.mm(R, torch.cat([x_corners, y_corners], dim=0))
                    corners_bev[0, :] = corners_bev[0, :] + center_x
                    corners_bev[1, :] = corners_bev[1, :] + center_y

                    debug = False
                    if i > 80 and j > 80 and debug:
                        bev_img = np.zeros((700, 800, 3))
                        print_corners = corners_bev.detach().numpy().T

                        print('Angle: ', int(angle / (2 * math.pi) * 360))
                        print('Length: ', length.item())
                        print('Width: ', width.item())
                        print('Pixel X: ', pixel_location_x_m)
                        print('Pixel Y: ', pixel_location_x_m)
                        print('Center X: ', center_x.item())
                        print('Center Y: ', center_y.item())

                        bev_img = kitti_utils.draw_projected_box_bev(bev_img, print_corners)
                        cv2.imshow('bev', bev_img)
                        cv2.waitKey()

                    corners_bev = corners_bev.flatten().unsqueeze(0)
                    pixel_prediction = (corners_bev, classification_prediction)
                    pixel_prediction = torch.cat(pixel_prediction, 1)

                    if final_prediction is None:
                        final_prediction = pixel_prediction
                    else:
                        final_prediction = torch.cat([final_prediction, pixel_prediction], dim=0)
                else:
                    continue

        # continue in case no positive predictions were made for entire sample
        if final_prediction is None:
            continue

        # get rid of all bounding boxes with object confidence below threshold (set to 0 before)
        non_zero_ind = (torch.nonzero(final_prediction[:, 8]))
        try:
            final_prediction = final_prediction[non_zero_ind.squeeze(), :].view(-1, 9)
        except:
            continue

        # For PyTorch 0.4 compatibility
        # Since the above code with not raise exception for no detection
        # as scalars are supported in PyTorch 0.4
        if final_prediction.shape[0] == 0:
            continue

        # sort the detections such that the entry with the maximum confidence score is at the top
        conf_sort_index = torch.sort(final_prediction[:, 8], descending=True)[1]
        final_prediction = final_prediction[conf_sort_index]
        n_detections = final_prediction.size(0)  # Number of detections
        # print('Number of positive detections before NMS: ', n_detections)

        # perform NMS
        for i in range(n_detections):
            # get the IOUs of all boxes that come after the one we are looking at
            try:
                ious = bbox_iou(final_prediction[i, :-1].unsqueeze(0), final_prediction[i + 1:, :-1])
            # break out of loop using exceptions due to dynamically changing tensor image_pred_class
            except ValueError:
                break
            except IndexError:
                break

            # zero out all the detections that have IoU > threshold
            iou_mask = (ious < nms_conf).float().unsqueeze(1)
            final_prediction[i + 1:] *= iou_mask

            # remove non-zero entries
            non_zero_ind = torch.nonzero(final_prediction[:, 0]).squeeze()
            final_prediction = final_prediction[non_zero_ind].view(-1, 9)

        batch_ind = torch.zeros([final_prediction.size(0), 1]).fill_(idp)
        # Repeat the batch_id for as many detections of the class cls in the image
        final_prediction = torch.cat([batch_ind, final_prediction], dim=1)

        print('Number of final detections: ', final_prediction.size(0))

        if final_predictions is None:
            final_predictions = final_prediction
        else:
            final_predictions = torch.cat([final_predictions, final_prediction])

    return final_predictions


###############
# Train Model #
###############


def evaluate_model(model, data_loader):

    model.eval()  # Set model to evaluate mode

    for batch_id, (batch_data, batch_labels, batch_calib) in enumerate(data_loader):

        with torch.set_grad_enabled(False):
            # forward pass
            batch_predictions = model(batch_data)
            batch_predictions = batch_predictions.permute([0, 2, 3, 1])

            # get final bounding box predictions
            output = process_predictions(batch_predictions)

            # display predictions and labels on BEV image
            batch_size = batch_data.size(0)
            for id in range(batch_size):
                # get point cloud as numpy array
                point_cloud = batch_data[id].detach().numpy().transpose((1, 2, 0))
                # draw BEV image
                bev_img = kitti_utils.draw_bev_image(point_cloud)

                # in case of valid predictions, draw bounding boxes on BEV image
                if output is not None:
                    # get predictions for currently inspected point cloud
                    predictions = torch.stack([prediction for prediction in output if prediction[0] == id], dim=0)
                    for idp in range(predictions.size(0)):
                        # get bbox corners
                        bbox_corners_camera_coord = predictions[idp, 1:-1].detach().numpy()
                        bbox_corners_camera_coord = np.reshape(bbox_corners_camera_coord, (2, 4)).T
                        # draw bbox on BEV image
                        bev_img = kitti_utils.draw_projected_box_bev(bev_img, bbox_corners_camera_coord, color=(0, 0, 255))

                # draw labels on BEV image
                for label in batch_labels[id]:
                    if label.type == 'Car':
                        # compute corners of the bounding box
                        bbox_corners_image_coord, bbox_corners_camera_coord = kitti_utils.compute_box_3d(label, batch_calib[id].P)
                        # draw BEV bounding box on BEV image
                        bev_img = kitti_utils.draw_projected_box_bev(bev_img, bbox_corners_camera_coord, color=(0, 255, 0))
                cv2.imshow('bev', bev_img)
                cv2.waitKey()


if __name__ == '__main__':
    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # evaluation parameters
    batch_size = 1

    # create data loader
    base_dir = 'kitti/'
    dataset_dir = 'object/'
    root_dir = os.path.join(base_dir, dataset_dir)
    data_loader = load_dataset(root=root_dir, batch_size=batch_size, device=device, test_set=True)

    # create model
    pixor = PIXOR()

    pixor.load_state_dict(torch.load('Models/PIXOR_Epoch_15.pt', map_location=device))

    # evaluate model
    evaluate_model(pixor, data_loader)
