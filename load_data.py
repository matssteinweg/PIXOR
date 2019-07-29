from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
import cv2
import kitti_utils
from config import *
import math
import time


#####################
# custom fc_collate #
#####################


def my_collate_test(batch):
    """
    Collate function for test dataset.
    How to concatenate individual samples to a batch.
    Point Clouds will be stacked along first dimension, labels and calibs will be returned as a list
    :param batch: list containing a tuple of items for each sample
    :return: batch data in desired form
    """

    point_clouds = []
    labels = []
    calibs = []
    for tuple_id, tuple in enumerate(batch):
        point_clouds.append(tuple[0])
        labels.append(tuple[1])
        calibs.append(tuple[2])

    point_clouds = torch.stack(point_clouds)
    return point_clouds, labels, calibs


def my_collate_train(batch):
    """
    Collate function for training dataset.
    How to concatenate individual samples to a batch.
    Point Clouds and labels will be stacked along first dimension
    :param batch: list containing a tuple of items for each sample
    :return: batch data in desired form
    """

    point_clouds = []
    labels = []
    for tuple_id, tuple in enumerate(batch):
        point_clouds.append(tuple[0])
        labels.append(tuple[1])

    point_clouds = torch.stack(point_clouds)
    labels = torch.stack(labels)
    return point_clouds, labels


########################
# compute pixel labels #
########################

def compute_pixel_labels(regression_label, classification_label, label, bbox_corners_camera_coord):
    """
    Compute the label that will be fed into the network from the bounding box annotations of the respective point cloud.
    :param: regression_label: emtpy numpy array | shape: [OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_REG]
    :param: classification_label: emtpy numpy array | shape: [OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_CLA]
    :param label: 3D label object containing bounding box information
    :param bbox_corners_camera_coord: corners of the bounding box | shape: [8, 3]
    :return: regression_label and classification_label filled with relevant label information
    """

    # get label information
    angle_rad = label.ry  # rotation of bounding box
    center_x_m = label.t[0]
    center_y_m = label.t[2]
    length_m = label.length
    width_m = label.width

    # extract corners of BEV bounding box
    bbox_corners_x = bbox_corners_camera_coord[:4, 0]
    bbox_corners_y = bbox_corners_camera_coord[:4, 2]

    # convert coordinates from m to pixels
    corners_x_px = ((bbox_corners_x - VOX_Y_MIN) // VOX_Y_DIVISION).astype(np.int32)
    corners_y_px = (INPUT_DIM_0 - ((bbox_corners_y - VOX_X_MIN) // VOX_X_DIVISION)).astype(np.int32)
    bbox_corners = np.vstack((corners_x_px, corners_y_px)).T

    # create a pixel mask of the target bounding box
    canvas = np.zeros((INPUT_DIM_0, INPUT_DIM_1, 3))
    canvas = cv2.fillPoly(canvas, pts=[bbox_corners], color=(255, 255, 255))

    # resize label to fit output shape
    canvas_resized = cv2.resize(canvas, (OUTPUT_DIM_1, OUTPUT_DIM_0), interpolation=cv2.INTER_NEAREST)
    bbox_mask = np.where(np.sum(canvas_resized, axis=2) == 765, 1, 0).astype(np.uint8)[:, :, np.newaxis]

    # get location of each pixel in m
    x_lin = np.linspace(VOX_Y_MIN, VOX_Y_MAX-0.4, OUTPUT_DIM_1)
    y_lin = np.linspace(VOX_X_MAX, VOX_X_MIN+0.4, OUTPUT_DIM_0)
    px_x, px_y = np.meshgrid(x_lin, y_lin)
    # create regression target
    target = np.array([[np.cos(angle_rad), np.sin(angle_rad), -center_x_m, -center_y_m, np.log(width_m), np.log(length_m)]])
    target = np.tile(target, (OUTPUT_DIM_0, OUTPUT_DIM_1, 1))
    # take offset from pixel as regression target for bounding box location
    target[:, :, 2] += px_x
    target[:, :, 3] += px_y
    # normalize target
    target = (target - REG_MEAN) / REG_STD
    # zero-out non-relevant pixels
    target *= bbox_mask

    regression_label += target
    classification_label += bbox_mask

    return regression_label, classification_label


###################
# dataset classes #
###################


class PointCloudDataset(Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, root_dir, split='training', device=torch.device('cpu'), show_times=True):
        """
        Dataset for training and testing containing point cloud, calibration object and in case of training labels
        :param root_dir: root directory of the dataset
        :param split: training or testing split of the dataset
        :param device: device on which dataset will be used
        :param show_times: show times of each step of the data loading (debug)
        """

        self.show_times = show_times  # debug

        self.device = device
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 6481
        elif split == 'testing':
            self.num_samples = 1000
        else:
            print('Unknown split: %s' % split)
            exit(-1)

        # paths to lidar, calibration and label directories
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.label_dir = os.path.join(self.split_dir, 'label_2')

    def __len__(self):
        # Denotes the total number of samples
        return self.num_samples

    def __getitem__(self, index):

        print('Index: ', index)

        # start time
        get_item_start_time = time.time()

        # get point cloud
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % index)
        lidar_data = kitti_utils.load_velo_scan(lidar_filename)

        # time for loading point cloud
        read_point_cloud_end_time = time.time()
        read_point_cloud_time = read_point_cloud_end_time - get_item_start_time

        # voxelize point cloud
        voxel_point_cloud = torch.tensor(kitti_utils.voxelize(point_cloud=lidar_data), requires_grad=True, device=self.device).float()

        # time for voxelization
        voxelization_end_time = time.time()
        voxelization_time = voxelization_end_time - read_point_cloud_end_time

        # channels along first dimensions according to PyTorch convention
        voxel_point_cloud = voxel_point_cloud.permute([2, 0, 1])

        # # random rotation between -5 and 5 degrees
        # angle = random.uniform(-5, 5) / 360 * (2 * math.pi)
        # c = np.cos(angle)
        # s = np.sin(angle)
        # R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        # lidar_data[:, :3] = np.dot(R, lidar_data[:, :3].T).T
        #
        # # horizontal flipping in 50% of the cases
        # flip = random.choice([0, 1])
        # if flip:
        #     lidar_data[:, 1] = -1 * lidar_data[:, 1]

        # create torch tensor from numpy array

        # get current time
        read_labels_start_time = time.time()

        # get calibration
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % index)
        calib = kitti_utils.Calibration(calib_filename)

        # get labels
        label_filename = os.path.join(self.label_dir, '%06d.txt' % index)
        labels = kitti_utils.read_label(label_filename)

        read_labels_end_time = time.time()
        read_labels_time = read_labels_end_time - read_labels_start_time

        # compute network label
        if self.split == 'training':
            # get current time
            compute_label_start_time = time.time()

            # create empty pixel labels
            regression_label = np.zeros((OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_REG))
            classification_label = np.zeros((OUTPUT_DIM_0, OUTPUT_DIM_1, OUTPUT_DIM_CLA))
            # iterate over all 3D label objects in list
            for label in labels:
                if label.type == 'Car':
                    # compute corners of 3D bounding box in camera coordinates
                    _, bbox_corners_camera_coord = kitti_utils.compute_box_3d(label, calib.P, scale=1.0)
                    # get pixel label for classification and BEV bounding box
                    regression_label, classification_label = compute_pixel_labels\
                        (regression_label, classification_label, label, bbox_corners_camera_coord)

            # stack classification and regression label
            regression_label = torch.tensor(regression_label, device=self.device).float()
            classification_label = torch.tensor(classification_label, device=self.device).float()
            training_label = torch.cat((regression_label, classification_label), dim=2)

            # get time for computing pixel label
            compute_label_end_time = time.time()
            compute_label_time = compute_label_end_time - compute_label_start_time

            # total time for data loading
            get_item_end_time = time.time()
            get_item_time = get_item_end_time - get_item_start_time

            if self.show_times:
                print('---------------------------')
                print('Get Item Time: {:.4f} s'.format(get_item_time))
                print('---------------------------')
                print('Read Point Cloud Time: {:.4f} s'.format(read_point_cloud_time))
                print('Voxelization Time: {:.4f} s'.format(voxelization_time))
                print('Read Labels Time: {:.4f} s'.format(read_labels_time))
                print('Compute Labels Time: {:.4f} s'.format(compute_label_time))

            return voxel_point_cloud, training_label

        else:

            return voxel_point_cloud, labels, calib


#################
# load datasets #
#################

def load_dataset(root='./kitti/object/', batch_size=1, train_val_split=0.9, test_set=False,
                 device=torch.device('cpu'), show_times=False):
    """
    Create a data loader that reads in the data from a directory of png-images
    :param device: device of the model
    :param root: root directory of the image data
    :param batch_size: batch-size for the data loader
    :param train_val_split: fraction of the available data used for training
    :param test_set: if True, data loader will be generated that contains only a test set
    :param show_times: display times for each step of the data loading
    :return: torch data loader object
    """

    # speed up data loading on gpu
    if device != torch.device('cpu'):
        num_workers = 0
    else:
        num_workers = 0

    # create training and validation set
    if not test_set:

        # create customized dataset class
        dataset = PointCloudDataset(root_dir=root, device=device, split='training', show_times=show_times)

        # number of images used for training and validation
        n_images = dataset.__len__()
        n_train = int(train_val_split * n_images)
        n_val = n_images - n_train

        # generated training and validation set
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

        # create data_loaders
        data_loader = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_train,
                                num_workers=num_workers),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_train,
                              num_workers=num_workers)
        }

    # create test set
    else:

        test_dataset = PointCloudDataset(root_dir=root, device=device, split='testing')
        data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_test,
                                 num_workers=num_workers, drop_last=True)

    return data_loader


if __name__ == '__main__':

    # create data loader
    base_dir = 'kitti/'
    dataset_dir = 'object/'
    root_dir = os.path.join(base_dir, dataset_dir)
    batch_size = 1
    device = torch.device('cpu')
    data_loader = load_dataset(root=root_dir, batch_size=batch_size, device=device, show_times=True)['train']

    for batch_id, (batch_data, batch_labels) in enumerate(data_loader):
        pass
