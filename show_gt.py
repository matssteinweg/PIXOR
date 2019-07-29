from load_data import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

##################
# dataset object #
##################


class KittiObject(object):
    """
    Load and parse object data into a usable format.

    """
    def __init__(self, root_dir, split='testing'):
        """
        root_dir contains training and testing folders
        :param root_dir:
        :param split:
        :param args:
        """
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 6481
        elif split == 'testing':
            self.num_samples = 1000
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert (idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png' % idx)
        return cv2.imread(img_filename)

    def get_calibration(self, idx):
        assert (idx < self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % idx)
        return kitti_utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert(idx < self.num_samples)
        label_filename = os.path.join(self.label_dir, '%06d.txt' % idx)
        return kitti_utils.read_label(label_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        assert (idx < self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))
        return kitti_utils.load_velo_scan(lidar_filename, dtype, n_vec)


########
# main #
########

if __name__ == '__main__':

    ###################
    # Executable: Yes #
    ###################

    """
    Explore the dataset. For a random selection of indices, the corresponding camera image will be displayed along with 
    all 3D bounding box annotations for the class "Cars". Moreover, the BEV image of the LiDAR point cloud will be 
    displayed with the bounding box annotations and a mask that shows the relevant pixels for the labels used for 
    training the network.
    """

    # root directory of the dataset
    base_dir = 'kitti/'
    dataset_dir = 'object/'
    root_dir = os.path.join(base_dir, dataset_dir)

    # create dataset
    train_dataset = KittiObject(root_dir)

    # select random indices from dataset
    ids = np.random.randint(0, 1000, 30)

    # loop over random selection
    for id in ids:

        # get image, point cloud, labels and calibration
        image = train_dataset.get_image(idx=id)
        labels = train_dataset.get_label_objects(idx=id)
        calib = train_dataset.get_calibration(idx=id)
        point_cloud = train_dataset.get_lidar(idx=id)

        # voxelize the point cloud
        voxel_point_cloud = kitti_utils.voxelize(point_cloud)

        # get BEV image of point cloud
        bev_image = kitti_utils.draw_bev_image(voxel_point_cloud)

        # create empty labels
        regression_label = np.zeros((OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_REG))
        classification_label = np.zeros((OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_CLA))

        # loop over all annotations for current sample
        for idl, label in enumerate(labels):
            # only display objects labeled as Car
            if label.type == 'Car':
                # compute corners of the bounding box
                bbox_corners_image_coord, bbox_corners_camera_coord = kitti_utils.compute_box_3d(label, calib.P)
                # draw BEV bounding box on BEV image
                bev_image = kitti_utils.draw_projected_box_bev(bev_image, bbox_corners_camera_coord)
                # create labels
                regression_label, classification_label = compute_pixel_labels(regression_label, classification_label,
                                                                   label, bbox_corners_camera_coord)
                # draw 3D bounding box on image
                if bbox_corners_image_coord is not None:
                    image = kitti_utils.draw_projected_box_3d(image, bbox_corners_image_coord)

        # create binary mask from relevant pixels in label
        label_mask = np.where(np.sum(np.abs(regression_label), axis=2) > 0, 255, 0).astype(np.uint8)

        idx = np.where(point_cloud[:, 0] > VOX_X_MIN)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 0] < VOX_X_MAX)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 1] > VOX_Y_MIN)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 1] < VOX_Y_MAX)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 2] > VOX_Z_MIN)
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:, 2] < VOX_Z_MAX)
        point_cloud = point_cloud[idx]

        # get rectified point cloud for depth information
        point_cloud_rect = calib.project_velo_to_rect(point_cloud[:, :3])

        # color map to indicate depth of point
        cmap = plt.cm.get_cmap('hsv', 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

        # project point cloud to image plane
        point_cloud_2d = calib.project_velo_to_image(point_cloud[:, :3]).astype(np.int32)

        # draw points
        for i in range(point_cloud_2d.shape[0]):
            depth = point_cloud_rect[i, 2]
            if depth > 0.1:
                color = cmap[int(255 - depth / VOX_X_MAX * 255)-1, :]
                cv2.circle(image, (point_cloud_2d[i, 0], point_cloud_2d[i, 1]), radius=2, color=color, thickness=-1)

        # display images
        cv2.imshow('Label Mask', label_mask)
        cv2.imshow('Image', image)
        cv2.imshow('Image_BEV', bev_image)
        cv2.waitKey()
