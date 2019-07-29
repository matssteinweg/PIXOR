import cv2
from config import *
import copy

###################
# 3D Label Object #
###################


class Object3D(object):
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.height = data[8]  # box height
        self.width = data[9]  # box width
        self.length = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.height, self.width, self.length))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))


######################
# Calibration Object #
######################


class Calibration(object):
    """
    Calibration matrices and utils

    ------------------
    coordinate systems
    ------------------
    3d XYZ in <label>.txt are in rect camera coord.
    2d box xy are in image2 coord
    Points in <lidar>.bin are in Velodyne coord.

    image2 coord:
     ----> x-axis (u)
    |
    |
    v y-axis (v)

    velodyne coord:
    front x, left y, up z

    rect/ref camera coord:
    right x, down y, front z

    ---------------------------
    camera -> image2 projection
    ---------------------------

        y_image2 = P2_rect * x_rect
        y_image2 = P2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P2_rect = [f2_u,  0,      c2_u,  -f2_u b2_x;
                    0,    f2_v,   c2_v,  -f2_v b2_y;
                    0,    0,      1,     0]
                 = K * [1|t]

    """

    def __init__(self, calib_filepath):

        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])
        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    @staticmethod
    def read_calib_file(filepath):
        """
        Read in a calibration file and parse into a dictionary.
        :param filepath:
        :return:
        """
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    ###############
    # projections #
    ###############

    @staticmethod
    def cart2hom(pts_3d):
        """
        Transform cartesian coordinates to homogeneous coordinates.
        :param pts_3d: nx3 points in Cartesian
        :return: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ---------
    # 3d to 3d
    # ---------

    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # --------
    # 3d to 2d
    # --------

    def project_rect_to_image(self, pts_3d_rect):
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)


###########################
# Compute 3D Bounding Box #
###########################


def compute_box_3d(label, P, scale=1.0):
    """
    Takes an object and a projection matrix (P) and projects the 3d bounding box into the image plane.
    :param label: 3d label object
    :param P: projection matrix
    :return: corners_2d: (8,2) array in left image coord.
             corners_3d: (8,3) array in in rect camera coord.
    """

    # compute rotational matrix around yaw axis
    R = rot_y(label.ry)

    # 3d bounding box dimensions in camera coordinates
    length = label.length * scale
    width = label.width * scale
    height = label.height * scale
    # 3d bounding box corners
    x_corners = [length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2]
    y_corners = [-height, -height, -height, -height, 0, 0, 0, 0]
    z_corners = [-width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + label.t[0]
    corners_3d[1, :] = corners_3d[1, :] + label.t[1]
    corners_3d[2, :] = corners_3d[2, :] + label.t[2]

    # only draw 3d bounding box for objects in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    return corners_2d, np.transpose(corners_3d)


#####################
# Drawing Functions #
#####################

# draw projected 3D bounding box
def draw_projected_box_3d(image, corners_2d, color=(0, 255, 0), thickness=2):
    """
    Draw 3d bounding box in image
            2 -------- 1
           /|         /|
          3 -------- 0 .
          | |        | |
          . 6 -------- 5
          |/         |/
          7 -------- 4
    :param image: image on which the bounding box will be drawn
    :param corners_2d: (8,2) array of vertices for the 3d box in following order
    :param color
    :param thickness
    :return: image with bounding box
    """

    corners_2d = corners_2d.astype(np.int32)

    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        image = cv2.line(image, (corners_2d[i, 0], corners_2d[i, 1]), (corners_2d[j, 0], corners_2d[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        image = cv2.line(image, (corners_2d[i, 0], corners_2d[i, 1]), (corners_2d[j, 0], corners_2d[j, 1]), color, thickness)
        i, j = k, k + 4
        image = cv2.line(image, (corners_2d[i, 0], corners_2d[i, 1]), (corners_2d[j, 0], corners_2d[j, 1]), color, thickness)

    return image


# draw projected BEV bounding box
def draw_projected_box_bev(image, corners_3d, color=(0, 255, 0), thickness=2):
    """
     Draw BEV bounding box on image
     :param image: bev image of observable area
     :param corners_3d: corners of 3D bounding box in camera coordinates
     :param color:
     :param thickness:
     :return: image with BEV bounding box
     """

    # extract corners of BEV bounding box
    if corners_3d.shape[1] == 3:
        corners_x = corners_3d[:4, 0]
        corners_y = corners_3d[:4, 2]
    else:
        corners_x = corners_3d[:4, 0]
        corners_y = corners_3d[:4, 1]

    # convert coordinates from m to image coordinates
    pixel_corners_x = ((corners_x - VOX_Y_MIN) // VOX_Y_DIVISION).astype(np.int32)
    pixel_corners_y = (INPUT_DIM_0 - ((corners_y - VOX_X_MIN) // VOX_X_DIVISION)).astype(np.int32)
    corners = np.vstack((pixel_corners_x, pixel_corners_y)).T

    # darker color for front line, lighter color inside
    color_inside = tuple([c - 55 if c == 255 else c for c in list(color)])
    color_front = tuple([c - 155 if c == 255 else c for c in list(color)])

    # draw polygon of bounding box on image
    # cv2.fillPoly(image, pts=[corners], color=color_inside)

    # draw BEV bounding box and mark front in different color
    image = cv2.line(image, (pixel_corners_x[3], pixel_corners_y[3]), (pixel_corners_x[0], pixel_corners_y[0]), color,
                     thickness)
    image = cv2.line(image, (pixel_corners_x[0], pixel_corners_y[0]), (pixel_corners_x[1], pixel_corners_y[1]), color_front,
                     thickness)
    image = cv2.line(image, (pixel_corners_x[1], pixel_corners_y[1]), (pixel_corners_x[2], pixel_corners_y[2]), color,
                     thickness)
    image = cv2.line(image, (pixel_corners_x[2], pixel_corners_y[2]), (pixel_corners_x[3], pixel_corners_y[3]), color,
                     thickness)

    return image


# draw BEV image
def draw_bev_image(point_cloud):
    """
    Generate a grayscale image displaying a BEV perspective of the point cloud
    :param point_cloud: voxelized point cloud | shape: [length, width, height]
    :return: numpy array | shape: [length, width, 3]
    """

    bev_image = np.sum(point_cloud, axis=2)
    bev_image = bev_image - np.min(bev_image)
    divisor = np.max(bev_image)-np.min(bev_image)
    bev_image = 255 - (bev_image/divisor*255)
    bev_image = np.dstack((bev_image, bev_image, bev_image)).astype(np.uint8)

    # add gray area to indicate camera FOV
    bev_image_bg = copy.copy(bev_image)
    triangle_cnt1 = np.array([(400, 700), (0, 700), (0, 300)])
    bev_image_bg = cv2.drawContours(bev_image_bg, [triangle_cnt1], 0, (120, 120, 120), -1)
    triangle_cnt2 = np.array([(400, 700), (800, 700), (800, 300)])
    bev_image_bg = cv2.drawContours(bev_image_bg, [triangle_cnt2], 0, (120, 120, 120), -1)

    # overlay background image to indicate FOV
    alpha = 0.2
    bev_image = cv2.addWeighted(bev_image_bg, alpha, bev_image, 1 - alpha, 0, bev_image)

    return bev_image


####################
# Reader Functions #
####################

# read label file
def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    lines = [line for line in lines if line.split(' ')[0] != 'DontCare']
    objects = [Object3D(line) for line in lines]
    return objects


# read lidar scan
def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan


####################
# Helper Functions #
####################

# invert transformation matrix
def inverse_rigid_trans(tr):
    """
    Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    :param tr: transformation matrix
    :return: inverse transformation matrix
    """
    inv_tr = np.zeros_like(tr)  # 3x4
    inv_tr[0:3, 0:3] = np.transpose(tr[0:3, 0:3])
    inv_tr[0:3, 3] = np.dot(-np.transpose(tr[0:3, 0:3]), tr[0:3, 3])
    return inv_tr


# rotation around y-axis
def rot_y(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


# project 3D points to image plane
def project_to_image(pts_3d, P):
    """
    Project 3d points to image plane.
    :param pts_3d: nx3 matrix
    :param P: 3x4 projection matrix
    :return: nx2 matrix
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


########################
# Voxelize Point Cloud #
########################


def voxelize(point_cloud):
    """
    Transform a continuous point cloud into a discrete voxelized grid that serves as the network input
    :param point_cloud: continuous point cloud | dim_0: all points, dim_1: [x, y, z, reflection]
    :return: voxelized point cloud | shape: [INPUT_DIM_0, INPUT_DIM_1, INPUT_DIM_2]
    """

    # remove all points outside the pre-specified FOV
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

    # create separate vectors for x, y, z coordinates and the reflectance value
    pxs = point_cloud[:, 0]
    pys = point_cloud[:, 1]
    pzs = point_cloud[:, 2]
    prs = point_cloud[:, 3]

    # convert velodyne coordinates to voxel
    qxs = (INPUT_DIM_0 - 1 - ((pxs - VOX_X_MIN) // VOX_X_DIVISION)).astype(np.int32)
    qys = ((-pys - VOX_Y_MIN) // VOX_Y_DIVISION).astype(np.int32)
    qzs = ((pzs - VOX_Z_MIN) // VOX_Z_DIVISION).astype(np.int32)
    quantized = np.dstack((qxs, qys, qzs, prs)).squeeze()

    # create empty voxel grid and reflectance image
    voxel_grid = np.zeros(shape=(INPUT_DIM_0, INPUT_DIM_1, INPUT_DIM_2-1), dtype=np.float32)
    reflectance_image = np.zeros(shape=(INPUT_DIM_0, INPUT_DIM_1), dtype=np.float32)
    reflectance_count = np.zeros(shape=(INPUT_DIM_0, INPUT_DIM_1), dtype=np.float32)

    # iterate over each point to fill occupancy grid and compute reflectance image
    for point_id, point in enumerate(quantized):
        point = point.astype(np.int32)
        voxel_grid[point[0], point[1], point[2]] = 1
        reflectance_image[point[0], point[1]] += point[3]
        reflectance_count[point[0], point[1]] += 1

    # take average over reflection of xy position
    reflectance_count = np.where(reflectance_count == 0, 1, reflectance_count)
    reflectance_image /= reflectance_count

    voxel_output = np.dstack((voxel_grid, reflectance_image))

    return voxel_output





