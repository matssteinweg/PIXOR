import torch.nn as nn
import torch
import copy
import os
import kitti_utils


###############
# Basis Block #
###############


class BasisBlock(nn.Module):
    """
    BasisBlock for input to ResNet
    """

    def __init__(self, n_input_channels):
        super(BasisBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


#################
# Residual Unit #
#################


class ResidualUnit(nn.Module):
    def __init__(self, n_input, n_output, downsample=False):
        """
        Residual Unit consisting of two convolutional layers and an identity mapping
        :param n_input: number of input channels
        :param n_output: number of output channels
        :param downsample: downsample the output by a factor of 2
        """
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output, n_output, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # down-sampling: use stride two for convolutional kernel and create 1x1 kernel for down-sampling of input
        self.downsample = None
        if downsample:
            self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.downsample = nn.Sequential(nn.Conv2d(n_input, n_output, kernel_size=(1, 1), stride=(2, 2), bias=False),
                                            nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True,
                                                           track_running_stats=True))
        else:
            self.identity_channels = nn.Conv2d(n_input, n_output, kernel_size=(1, 1), bias=False)

    def forward(self, x):

        # store input for skip-connection
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # downsample input to match output dimensions
        if self.downsample is not None:
            identity = self.downsample(identity)
        else:
            identity = self.identity_channels(identity)

        # skip-connection
        x += identity

        # apply ReLU activation
        x = self.relu(x)

        return x


##################
# Residual Block #
##################


class ResidualBlock(nn.Module):
    """
        Residual Block containing specified number of residual layers
        """

    def __init__(self, n_input, n_output, n_res_units):
        super(ResidualBlock, self).__init__()

        # use down-sampling only in the first residual layer of the block
        first_unit = True

        # specific channel numbers
        if n_res_units == 3:
            inputs = [n_input, n_output//4, n_output//4]
            outputs = [n_output//4, n_output//4, n_output]
        else:
            inputs = [n_input, n_output // 4, n_output // 4, n_output // 4, n_output // 4, n_output]
            outputs = [n_output // 4, n_output // 4, n_output // 4, n_output // 4, n_output, n_output]

        # create residual units
        units = []
        for unit_id in range(n_res_units):
            if first_unit:
                units.append(ResidualUnit(inputs[unit_id], outputs[unit_id], downsample=True))
                first_unit = False
            else:
                units.append(ResidualUnit(inputs[unit_id], outputs[unit_id]))
        self.res_block = nn.Sequential(*units)

    def forward(self, x):

        x = self.res_block(x)

        return x


#############
# FPN Block #
#############


class FPNBlock(nn.Module):
    """
        Block for Feature Pyramid Network including up-sampling and concatenation of feature maps
        """

    def __init__(self, bottom_up_channels, top_down_channels, fused_channels):
        super(FPNBlock, self).__init__()
        # reduce number of top-down channels to 196
        intermediate_channels = 196
        if top_down_channels > 196:
            self.channel_conv_td = nn.Conv2d(top_down_channels, intermediate_channels, kernel_size=(1, 1),
                                             stride=(1, 1), bias=False)
        else:
            self.channel_conv_td = None

        # change number of bottom-up channels to 128
        self.channel_conv_bu = nn.Conv2d(bottom_up_channels, fused_channels, kernel_size=(1, 1),
                                         stride=(1, 1), bias=False)

        # transposed convolution on top-down feature maps
        if fused_channels == 128:
            out_pad = (1, 1)
        else:
            out_pad = (0, 1)
        if self.channel_conv_td is not None:
            self.deconv = nn.ConvTranspose2d(intermediate_channels, fused_channels, kernel_size=(3, 3), padding=(1, 1),
                                             stride=2, output_padding=out_pad)
        else:
            self.deconv = nn.ConvTranspose2d(top_down_channels, fused_channels, kernel_size=(3, 3), padding=(1, 1),
                                             stride=2, output_padding=out_pad)

    def forward(self, x_td, x_bu):

        # apply 1x1 convolutional to obtain required number of channels if needed
        if self.channel_conv_td is not None:
            x_td = self.channel_conv_td(x_td)

        # up-sample top-down feature maps
        x_td = self.deconv(x_td)

        # apply 1x1 convolutional to obtain required number of channels
        x_bu = self.channel_conv_bu(x_bu)

        # perform element-wise addition
        x = x_td.add(x_bu)

        return x


####################
# Detection Header #
####################

class DetectionHeader(nn.Module):

    def __init__(self, n_input, n_output):
        super(DetectionHeader, self).__init__()
        basic_block = nn.Sequential(nn.Conv2d(n_input, n_output, kernel_size=(3, 3), padding=(1, 1), bias=False),
                                  nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True))
        self.conv1 = basic_block
        self.conv2 = copy.deepcopy(basic_block)
        self.conv3 = copy.deepcopy(basic_block)
        self.conv4 = copy.deepcopy(basic_block)
        self.classification = nn.Conv2d(n_output, 1, kernel_size=(3, 3), padding=(1, 1))
        self.regression = nn.Conv2d(n_output, 6, kernel_size=(3, 3), padding=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        class_output = self.sigmoid(self.classification(x))
        regression_output = self.regression(x)

        return class_output, regression_output


#########
# PIXOR #
#########


class PIXOR(nn.Module):
    def __init__(self):
        super(PIXOR, self).__init__()

        # Backbone Network
        self.basis_block = BasisBlock(n_input_channels=36)
        self.res_block_1 = ResidualBlock(n_input=32, n_output=96, n_res_units=3)
        self.res_block_2 = ResidualBlock(n_input=96, n_output=196, n_res_units=6)
        self.res_block_3 = ResidualBlock(n_input=196, n_output=256, n_res_units=6)
        self.res_block_4 = ResidualBlock(n_input=256, n_output=384, n_res_units=3)

        # FPN blocks
        self.fpn_block_1 = FPNBlock(top_down_channels=384, bottom_up_channels=256, fused_channels=128)
        self.fpn_block_2 = FPNBlock(top_down_channels=128, bottom_up_channels=196, fused_channels=96)

        # Detection Header
        self.header = DetectionHeader(n_input=96, n_output=96)

    def forward(self, x):
        x_b = self.basis_block(x)
        # print(x_b.size())
        x_1 = self.res_block_1(x_b)
        # print(x_1.size())
        x_2 = self.res_block_2(x_1)
        # print(x_2.size())
        x_3 = self.res_block_3(x_2)
        # print(x_3.size())
        x_4 = self.res_block_4(x_3)
        # print(x_4.size())
        x_34 = self.fpn_block_1(x_4, x_3)
        # print(x_34.size())
        x_234 = self.fpn_block_2(x_34, x_2)
        # print(x_234.size())
        x_class, x_reg = self.header(x_234)
        # print(x_class.size())
        # print(x_reg.size())
        x_out = torch.cat((x_reg, x_class), dim=1)

        return x_out


########
# Main #
########


if __name__ == '__main__':

    # exemplary input point cloud
    base_dir = 'Data/training/velodyne'
    index = 1
    lidar_filename = os.path.join(base_dir, '%06d.bin' % index)
    lidar_data = kitti_utils.load_velo_scan(lidar_filename)
    # create torch tensor from numpy array
    voxel_point_cloud = torch.tensor(kitti_utils.voxelize(lidar_data), requires_grad=True, device='cpu').float()
    # channels along first dimensions according to PyTorch convention
    voxel_point_cloud = voxel_point_cloud.permute([2, 0, 1])
    voxel_point_cloud = torch.unsqueeze(voxel_point_cloud, 0)  # add dimension 0 to tensor for batch

    # forward pass through network
    pixor = PIXOR()
    prediction = pixor(voxel_point_cloud)
    classification_prediction = prediction[:, :, -1]
    regression_prediction = prediction[:, :, :-1]

    print('+++++++++++++++++++++++++++++++++++++')
    print('BEV Backbone Network')
    print('+++++++++++++++++++++++++++++++++++++')
    print(pixor)
    print('+++++++++++++++++++++++++++++++++++++')

    for child_name, child in pixor.named_children():
        print('++++++++++++++++++++++')
        print(child_name)
        print('++++++++++++++++++++++')
        for parameter_name, parameter in child.named_parameters():
                print(parameter_name)

