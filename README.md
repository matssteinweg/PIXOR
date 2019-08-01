# PIXOR: Real-time 3D Object Detection from Point Clouds
## Unofficial PyTorch Implementation

In this repository you'll find an unofficial implementation of PIXOR using PyTorch. I implemented this project to gain some experience working with 3D object detection and familiarize myself with the *kitti dataset* used for training and evaluation of the model. The vast majority of the code presented in this repository is written by myself based on my interpretation of the [original paper](https://arxiv.org/pdf/1902.06326.pdf). Parts of the helper functions for loading and displaying data in ```kitti_utils.py``` are inspired by the [kitti_object_vis](https://github.com/kuixu/kitti_object_vis) repository.

### Requirements

The project is built using a small set of libraries. Post-processing is predominantly performed in numpy with scipy being used for some vectorized operations. Shapely is required for the calculation of bounding box IoUs. All image modifications are performed using OpenCV while matplotlib is used for plotting of training history and evaluation plots.
```
torch 1.1.0
shapely
scipy
cv2
matplotlib
numpy
```

### Dataset

The kitti dataset is among the most widely used datasets for the development of computer vision and sensor fusion applications. The dataset can be downloaded from the [official website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=bev) for free. For this project I only used the LiDAR point clouds. However, I downloaded the camera images as well to get a better idea of the car's surroundings during the dataset exploration. Furthermore, I downloaded the camera calibration matrices and the bounding box annotations. 
The project requires the dataset to be save according to the following structure:
```
Data
  training
     velodyne
     calib
     image_2
     label_2
  testing
     velodyne
     calib
     image_2
     label_2
 ```
 The training set consists of 7481 annotated frames. There also exists an official test set containing 7518 frames. However, the labels are only available for implementations that are associated with a published paper. Consequently, I had to rely on a fraction of the original training set for the evaluation of my model. I split the available data into a training set of 6481 frames and a testing set of 1000 samples. During training the dataset is then further split into a training and a validation set using a 90/10 split.
 
### Implementation Details

The implementation and evaluation of the model is aimed to reproduce the approach taken in the original paper. This refers to the basic implementation details provided in the paper. Modifications discussed in the ablation study are not included. 
A difference worth mentioning is that the model was trained in a binary classification manner. Instead of considering all available classes of the kitti dataset (Car, Van, Truck, Pedestrian, Person (sitting), Cyclist, Tram and Misc) the model is merely trained on annotations of the class Car. This is expected to be updated in the future to a model trained on all classes.

### How-To Navigate the Project

With the folder structure for the dataset set up according to the specifications above, you're ready to navigate the project.
The project directory contains three main files of interest, one for training, evaluation and detection, respectively.

#### Training
First, we want to train a new PIXOR model. The training is performed in ```train_model.py```. By running the script a new model is trained using the specified training parameters. I trained the model for 30 epochs using Adam with an initial learning rate of 0.001 and a scheduler that reduced the learning rate by a factor of 0.1 after 10 and 20 epochs, respectively. Early stopping is used with a default patience of 8 epochs. The batch size is set 6 due to memory constraints of the GPU I used for training. During training, the model is saved after each epoch in case the validation loss decreased compared to the previous epoch. In order to save the model, a folder ```Models``` has to be created in the working directory. Furthermore, a folder ```Metrics```should be created. In this folder, a dictionary containing the training and validation loss is stored after every epoch to allow for visualizing the training process.

#### Evaluation
With a PIXOR model trained and saved to the Models folder of the working directory, the model can be evaluated on the test set. Sticking to the evaluation scheme of the original paper, the performance is measured over three different distance ranges(0-30m, 0-50m and 0-70m) and the mAP is computed as an average over IoU thresholds of 0.5, 0.6, 0.7, 0.8 and 0.9.
By running ```evaluate_model.py```, a dictionary containing all relevant performance measures is created and saved. Prior to the execution a folder ```Eval```has to be created in the working directory. Having created the evaluation dictionary, the evaluation can be visualized using ```visualize_evaluation.py```. For each of the evaluated distance ranges, Precision-Recall-Curves are plotted for each of the specified IoU thresholds. Moreover, the final mAP for each distance range is displayed.

<p align="center"> 
<img height="400px" src="/Images/pr_curve_epoch_17.png">
</p>


#### Detection
Having trained a PIXOR model, the detector can be run on unseen point clouds. For an visual inspection of the resulting detections, run ```detector.py```. In this script, the detector is run on a set of selected indices from the test set and the results are displayed. In order to get a good intuition about the quality of the results, the detections are displayed on a BEV representation of the point cloud along with the ground truth bounding boxes. Furthermore, an option exists to also display the original camera image along with projections of the predicted and annotated bounding boxes.



