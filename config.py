import numpy as np

###########################
# project-level constants #
###########################

# observable area in m in velodyne coordinates
VOX_Y_MIN = -40
VOX_Y_MAX = +40
VOX_X_MIN = 0
VOX_X_MAX = 70
VOX_Z_MIN = -2.5
VOX_Z_MAX = 1.0

# transformation from m to voxels
VOX_X_DIVISION = 0.1
VOX_Y_DIVISION = 0.1
VOX_Z_DIVISION = 0.1

# dimensionality of network input (voxelized point cloud)
INPUT_DIM_0 = int((VOX_X_MAX-VOX_X_MIN) // VOX_X_DIVISION) + 1
INPUT_DIM_1 = int((VOX_Y_MAX-VOX_Y_MIN) // VOX_Y_DIVISION) + 1
# + 1 for average reflectance value of the points in the respective voxel
INPUT_DIM_2 = int((VOX_Z_MAX-VOX_Z_MIN) // VOX_Z_DIVISION) + 1 + 1

# dimensionality of network output
OUTPUT_DIM_0 = INPUT_DIM_0 // 4
OUTPUT_DIM_1 = INPUT_DIM_1 // 4
OUTPUT_DIM_REG = 6
OUTPUT_DIM_CLA = 1

# mean and std for normalization of the regression targets
REG_MEAN = np.array([-0.01518276, -0.0626486, -0.05025632, -0.05040792, 0.49188597, 1.36500531])
REG_STD = np.array([0.46370442, 0.88364181, 0.70925018, 1.0590797, 0.06251486, 0.10906765])
