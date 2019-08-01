from evaluate_model import *

###############
# show legend #
###############


def show_legend(image, ground_truth_color, prediction_color, idx):
    """
    Display legend for color codes in provided image.
    :param image: image on which to display the legend
    :return: image with legend
    """

    text1 = 'Ground Truth'
    text2 = 'Prediction'
    text3 = 'Point Cloud Index: {:d}'.format(idx)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7
    thickness = 1
    size1 = cv2.getTextSize(text1, font, font_scale, thickness)
    size2 = cv2.getTextSize(text2, font, font_scale, thickness)
    size3 = cv2.getTextSize(text3, font, font_scale, thickness)
    rectangle = cv2.rectangle(image, (20, 20),
                              (20 + max(size1[0][0], size2[0][0], size3[0][0]) + 10,
                               20 + size1[0][1] + size2[0][1] + size3[0][1] + 20),
                              (200, 200, 200), 1)
    cv2.putText(rectangle, text3, (25, 25 + size3[0][1]), font, font_scale, (100, 100, 100), 1)
    cv2.putText(rectangle, text1, (25, 30 + size1[0][1] + size3[0][1]), font, font_scale, ground_truth_color, 1)
    cv2.putText(rectangle, text2, (25, 35 + size1[0][1] + size2[0][1] + size3[0][1]), font, font_scale, prediction_color, 1)

    return image


############
# detector #
############

if __name__ == '__main__':

    """
    Run the detector. Iterate over a set of indices and display final detections for the respective point cloud
    in a BEV image and the original camera image.
    """

    # root directory of the dataset
    root_dir = 'Data/'

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create dataset
    dataset = PointCloudDataset(root_dir, split='testing', get_image=True)

    # select index from dataset
    ids = np.arange(0, dataset.__len__())

    for id in ids:

        # get image, point cloud, labels and calibration
        camera_image, point_cloud, labels, calib = dataset.__getitem__(id)

        # create model
        pixor = PIXOR()
        n_epochs_trained = 17
        pixor.load_state_dict(torch.load('Models/PIXOR_Epoch_' + str(n_epochs_trained) + '.pt', map_location=device))

        # unsqueeze first dimension for batch
        point_cloud = point_cloud.unsqueeze(0)

        # forward pass
        predictions = pixor(point_cloud)

        # convert network output to numpy for further processing
        predictions = np.transpose(predictions.detach().numpy(), (0, 2, 3, 1))

        # get final bounding box predictions
        final_box_predictions = process_predictions(predictions, confidence_threshold=0.5)

        ###################
        # display results #
        ###################

        # set colors
        ground_truth_color = (80, 127, 255)
        prediction_color = (255, 127, 80)

        # get point cloud as numpy array
        point_cloud = point_cloud[0].detach().numpy().transpose((1, 2, 0))

        # draw BEV image
        bev_image = kitti_utils.draw_bev_image(point_cloud)

        # display ground truth bounding boxes on BEV image and camera image
        for label in labels:
            # only consider annotations for class "Car"
            if label.type == 'Car':
                # compute corners of the bounding box
                bbox_corners_image_coord, bbox_corners_camera_coord = kitti_utils.compute_box_3d(label, calib.P)
                # display bounding box in BEV image
                bev_img = kitti_utils.draw_projected_box_bev(bev_image, bbox_corners_camera_coord, color=ground_truth_color)
                # display bounding box in camera image
                if bbox_corners_image_coord is not None:
                    camera_image = kitti_utils.draw_projected_box_3d(camera_image, bbox_corners_image_coord, color=ground_truth_color)

        # display predicted bounding boxes on BEV image and camera image
        if final_box_predictions is not None:
            for prediction in final_box_predictions:
                bbox_corners_camera_coord = np.reshape(prediction[2:], (2, 4)).T
                # create 3D bounding box coordinates from BEV coordinates. Place all bounding boxes on the ground and
                # choose a height of 1.5m
                bbox_corners_camera_coord = np.tile(bbox_corners_camera_coord, (2, 1))
                bbox_y_camera_coord = np.array([[0., 0., 0., 0., 1.65, 1.65, 1.65, 1.65]]).T
                bbox_corners_camera_coord = np.hstack((bbox_corners_camera_coord, bbox_y_camera_coord))
                switch_indices = np.argsort([0, 2, 1])
                bbox_corners_camera_coord = bbox_corners_camera_coord[:, switch_indices]
                bbox_corners_image_coord = kitti_utils.project_to_image(bbox_corners_camera_coord, calib.P)

                # display bounding box with confidence score in BEV image
                bev_img = kitti_utils.draw_projected_box_bev(bev_image, bbox_corners_camera_coord, color=prediction_color, confidence_score=prediction[1])
                # display bounding box in camera image
                if bbox_corners_image_coord is not None:
                    camera_image = kitti_utils.draw_projected_box_3d(camera_image, bbox_corners_image_coord, color=prediction_color)

        # display legend on BEV Image
        bev_image = show_legend(bev_image, ground_truth_color, prediction_color, id)

        # show images
        # cv2.imshow('BEV Image', bev_image)
        # cv2.imshow('Camera Image', camera_image)
        # cv2.waitKey()

        # save image
        print('Index: ', id)
        cv2.imwrite('Images/Detections/detection_id_{:d}.png'.format(id), bev_image)
