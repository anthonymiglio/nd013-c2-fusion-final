
# SDCND : Sensor Fusion and Tracking
This is the project for the second course in the  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213) : Sensor Fusion and Tracking. 

In this project, you'll fuse measurements from LiDAR and camera and track vehicles over time. You will be using real-world data from the Waymo Open Dataset, detect objects in 3D point clouds and apply an extended Kalman filter for sensor fusion and tracking.

<img src="img/img_title_1.jpeg"/>

The project consists of two major parts: 
1. **Object detection**: In this part, a deep-learning approach is used to detect vehicles in LiDAR data based on a birds-eye view perspective of the 3D point-cloud. Also, a series of performance measures is used to evaluate the performance of the detection approach. 
2. **Object tracking** : In this part, an extended Kalman filter is used to track vehicles over time, based on the lidar detections fused with camera detections. Data association and track management are implemented as well.

The following diagram contains an outline of the data flow and of the individual steps that make up the algorithm. 

<img src="img/img_title_2_new.png"/>

Also, the project code contains various tasks, which are detailed step-by-step in the code. More information on the algorithm and on the tasks can be found in the Udacity classroom. 

## Project File Structure

📦project<br>
 ┣ 📂dataset --> contains the Waymo Open Dataset sequences <br>
 ┃<br>
 ┣ 📂misc<br>
 ┃ ┣ evaluation.py --> plot functions for tracking visualization and RMSE calculation<br>
 ┃ ┣ helpers.py --> misc. helper functions, e.g. for loading / saving binary files<br>
 ┃ ┗ objdet_tools.py --> object detection functions without student tasks<br>
 ┃ ┗ params.py --> parameter file for the tracking part<br>
 ┃ <br>
 ┣ 📂results --> binary files with pre-computed intermediate results<br>
 ┃ <br>
 ┣ 📂student <br>
 ┃ ┣ association.py --> data association logic for assigning measurements to tracks incl. student tasks <br>
 ┃ ┣ filter.py --> extended Kalman filter implementation incl. student tasks <br>
 ┃ ┣ measurements.py --> sensor and measurement classes for camera and lidar incl. student tasks <br>
 ┃ ┣ objdet_detect.py --> model-based object detection incl. student tasks <br>
 ┃ ┣ objdet_eval.py --> performance assessment for object detection incl. student tasks <br>
 ┃ ┣ objdet_pcl.py --> point-cloud functions, e.g. for birds-eye view incl. student tasks <br>
 ┃ ┗ trackmanagement.py --> track and track management classes incl. student tasks  <br>
 ┃ <br>
 ┣ 📂tools --> external tools<br>
 ┃ ┣ 📂objdet_models --> models for object detection<br>
 ┃ ┃ ┃<br>
 ┃ ┃ ┣ 📂darknet<br>
 ┃ ┃ ┃ ┣ 📂config<br>
 ┃ ┃ ┃ ┣ 📂models --> darknet / yolo model class and tools<br>
 ┃ ┃ ┃ ┣ 📂pretrained --> copy pre-trained model file here<br>
 ┃ ┃ ┃ ┃ ┗ complex_yolov4_mse_loss.pth<br>
 ┃ ┃ ┃ ┣ 📂utils --> various helper functions<br>
 ┃ ┃ ┃<br>
 ┃ ┃ ┗ 📂resnet<br>
 ┃ ┃ ┃ ┣ 📂models --> fpn_resnet model class and tools<br>
 ┃ ┃ ┃ ┣ 📂pretrained --> copy pre-trained model file here <br>
 ┃ ┃ ┃ ┃ ┗ fpn_resnet_18_epoch_300.pth <br>
 ┃ ┃ ┃ ┣ 📂utils --> various helper functions<br>
 ┃ ┃ ┃<br>
 ┃ ┗ 📂waymo_reader --> functions for light-weight loading of Waymo sequences<br>
 ┃<br>
 ┣ basic_loop.py<br>
 ┣ loop_over_dataset.py<br>



## Installation Instructions for Running Locally
### Cloning the Project
In order to create a local copy of the project, please click on "Code" and then "Download ZIP". Alternatively, you may of-course use GitHub Desktop or Git Bash for this purpose. 

### Python
The project has been written using Python 3.7. Please make sure that your local installation is equal or above this version. 

### Package Requirements
All dependencies required for the project have been listed in the file `requirements.txt`. You may either install them one-by-one using pip or you can use the following command to install them all at once: 
`pip3 install -r requirements.txt` 

### Waymo Open Dataset Reader
The Waymo Open Dataset Reader is a very convenient toolbox that allows you to access sequences from the Waymo Open Dataset without the need of installing all of the heavy-weight dependencies that come along with the official toolbox. The installation instructions can be found in `tools/waymo_reader/README.md`. 

### Waymo Open Dataset Files
This project makes use of three different sequences to illustrate the concepts of object detection and tracking. These are: 
- Sequence 1 : `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`
- Sequence 2 : `training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord`
- Sequence 3 : `training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord`

To download these files, you will have to register with Waymo Open Dataset first: [Open Dataset – Waymo](https://waymo.com/open/terms), if you have not already, making sure to note "Udacity" as your institution.

Once you have done so, please [click here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files) to access the Google Cloud Container that holds all the sequences. Once you have been cleared for access by Waymo (which might take up to 48 hours), you can download the individual sequences. 

The sequences listed above can be found in the folder "training". Please download them and put the `tfrecord`-files into the `dataset` folder of this project.


### Pre-Trained Models
The object detection methods used in this project use pre-trained models which have been provided by the original authors. They can be downloaded [here](https://drive.google.com/file/d/1Pqx7sShlqKSGmvshTYbNDcUEYyZwfn3A/view?usp=sharing) (darknet) and [here](https://drive.google.com/file/d/1RcEfUIF1pzDZco8PJkZ10OL-wLL2usEj/view?usp=sharing) (fpn_resnet). Once downloaded, please copy the model files into the paths `/tools/objdet_models/darknet/pretrained` and `/tools/objdet_models/fpn_resnet/pretrained` respectively.

### Using Pre-Computed Results

In the main file `loop_over_dataset.py`, you can choose which steps of the algorithm should be executed. If you want to call a specific function, you simply need to add the corresponding string literal to one of the following lists: 

- `exec_data` : controls the execution of steps related to sensor data. 
  - `pcl_from_rangeimage` transforms the Waymo Open Data range image into a 3D point-cloud
  - `load_image` returns the image of the front camera

- `exec_detection` : controls which steps of model-based 3D object detection are performed
  - `bev_from_pcl` transforms the point-cloud into a fixed-size birds-eye view perspective
  - `detect_objects` executes the actual detection and returns a set of objects (only vehicles) 
  - `validate_object_labels` decides which ground-truth labels should be considered (e.g. based on difficulty or visibility)
  - `measure_detection_performance` contains methods to evaluate detection performance for a single frame

In case you do not include a specific step into the list, pre-computed binary files will be loaded instead. This enables you to run the algorithm and look at the results even without having implemented anything yet. The pre-computed results for the mid-term project need to be loaded using [this](https://drive.google.com/drive/folders/1-s46dKSrtx8rrNwnObGbly2nO3i4D7r7?usp=sharing) link. Please use the folder `darknet` first. Unzip the file within and put its content into the folder `results`.

- `exec_tracking` : controls the execution of the object tracking algorithm

- `exec_visualization` : controls the visualization of results
  - `show_range_image` displays two LiDAR range image channels (range and intensity)
  - `show_labels_in_image` projects ground-truth boxes into the front camera image
  - `show_objects_and_labels_in_bev` projects detected objects and label boxes into the birds-eye view
  - `show_objects_in_bev_labels_in_camera` displays a stacked view with labels inside the camera image on top and the birds-eye view with detected objects on the bottom
  - `show_tracks` displays the tracking results
  - `show_detection_performance` displays the performance evaluation based on all detected 
  - `make_tracking_movie` renders an output movie of the object tracking results

Even without solving any of the tasks, the project code can be executed. 

The final project uses pre-computed lidar detections in order for all students to have the same input data. If you use the workspace, the data is prepared there already. Otherwise, [download the pre-computed lidar detections](https://drive.google.com/drive/folders/1IkqFGYTF6Fh_d8J3UjQOSNJ2V42UDZpO?usp=sharing) (~1 GB), unzip them and put them in the folder `results`.

## External Dependencies
Parts of this project are based on the following repositories: 
- [Simple Waymo Open Dataset Reader](https://github.com/gdlg/simple-waymo-open-dataset-reader)
- [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D)
- [Complex-YOLO: Real-time 3D Object Detection on Point Clouds](https://github.com/maudzung/Complex-YOLOv4-Pytorch)


## License
[License](LICENSE.md)


# Writeup: Track 3D-Objects Over Time

After completing the final project, we will have implemented a sensor fusion system that can track vehicles over time with a real-world camera and lidar measurements!


## The final project consists of four main steps: Filter, Track Management, Association, Camera Fusion.

And some further analysis:
## 1. Which part of the project was most difficult for you to complete, and why?
## 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 
## 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?
## 4. Can you think of ways to improve your tracking results in the future?

To run this project in the workspace folder: 
```
python loop_over_dataset.py
```
And in Set parameters and perform initializations, select an exercise to run: 'step1', 'step2', 'step3', 'step4'


**Here is a brief recap of the steps:**

### Step 1: Implement an Extended Kalman Filter.
Step 1 of the final project is an Extended Kalman Filter (EKF) implementation to track a real-world single-target scenario with lidar measurement input over time. 
In the student/filter.py file, There are two steps: predict and measure. The Prediction step predicts x and predicts P based on the motion model. The Measurement step is to update x and update P based on the measurement error and the covariances.

Implemented functions:
- *F* to calculate a system matrix for a constant velocity motion model in 3D, and *Q* the process noise covariance depending on the current timestep dt. This applies to a function *predict* to predict state x and estimation error covariance P to the next timestep, saving x and P in track.
- Calculate *gamma* as the residual and *S* as the covariance of the residual. This applies to a function *update* to update state x and covariance P with associated measurement, saving x and P in track.

Result: The image below shows the analysis of RMSE for a single tracking.
<img src="/img/step1_rmse_single_target_tracking.png"/>


### Step 2: Implement a Track Management
Step 2 of the final project is implementing track management, calculating the track score, and switching between initialized, tentative, and confirmed track states.

**File:** trackmanagement.py
- First, initialize the track with an unassigned lidar calculation;
- Track state is defined according to track score;
- If the track score is correlated with measurement, the corresponding scores will be increased, or if not, the track score will decrease.
- Old tracks are deleted for not updated tracks.
- If the track state is confirmed and track score is below a certain threshold, or track state is tentative and covariance P of position x or position y is greater than a maximum limit, then the track is not removed for further consideration.

Result: The image below shows the analysis of RMSE for a single tracking.
<img src="/img/step2_rmse_single_target_tracking.png"/>


### Step 3: Implement SNN Data Association and Gating
The third step is to implement the association of measurements to tracks and to handle unassociated tracks and measurements. We use a single nearest neighbor data association measured by Mahalanobis distance and use gating to ease associations.

**File:** association.py
- Create an association matrix with N tracks by M measurements;
- Initialize association matrix with infinite values;
- Loop over all tracks and all measurements to set up an association matrix:
1. Calculate *MHD* (Mahalanobis distance) between the track and measurement;
2. Check if the measurement lies inside the gate; if not, exclude unlikely track pairs. For this, use the hypothesis test Chi-Square.
3. The smallest MHD is chosen, update Kalman Filter, and delete the selected pair row and column from the association matrix until no more assigned pairs.

Result: The image below shows the analysis of RMSE for SNN Data Association:
<img src="/img/step3_rmse_lidar.png"/>


### Step 4: Apply sensor fusion by implementing the nonlinear camera measurement model and a sensor visibility check.
The fourth step is to implement camera-lidar fusion, making the extended Kalman filter support the nonlinear transformation of the camera measurement and linear for lidar measurement. The projection matrix converts real-world 3D points to picture 2D points. Use partial derivatives of x, y, and z to measure the model in u,v parameters.

**File:** measurements.py
Implemented functions:
- *in_fov* checks whether an object x lies in the sensor's field of view. 
- *get_hx* calculates the nonlinear measurement expectation value h(x) for the camera sensor.
- *generate_measurement* creates a new measurement from this sensor and adds it to the measurement list.
- and the *Measurement* class, create a measurement object that initializes the camera measurement of vector z, noise covariance matrix R, for the camera sensor.

Result: The video below shows frame-by-frame the camera-lidar fusion sensor front view and BEF(Birds Eye View)
<img src="/results/my_tracking_results.gif">



### 1. Which part of the project was most difficult for you to complete, and why?
The lectures helped as a guide to implementing the four steps: EKF, track management, data association, and camera-lidar fusion. Nonetheless, it was challenging to implement the camera measuring model with its transformations in the camera axis to project a world 3D point into an image 2D point. As well as calculating the Jacobian matrix 


### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)? 
The benefit of sensor fusion is combining multiple sensors and gaining the best features from different sensors, especially if they are from various natures.
Cameras may offer color, brightness, and contrast, while a Lidar is highly advantageous in low light and bad weather conditions such as fog or rain. Including the camera, fusion tracking produces a better geometric project matrix that is good for the sensors to work with.

The picture below is a sensor fusion with lidar and camera detection, and all tracks have lower ERMS than the lidar-only image in Step 3.
<img src="/img/step4_rmse_sensorfusion.png"/>


### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?
In real-life scenarios:
- Periodic sensor calibration is crucial for precise coordinate translation from the sensor to the vehicle.
- The unpredictability of the weather conditions might impact sensor performance in situations such as fog, heavy rain, direct sun rays, etc.
- Heavy traffic increases the number of cars on the road, while light traffic increases the need for a reaction time for breaking or deviating.


### 4. Can you think of ways to improve your tracking results in the future?
- Increase frame rate: reduce estimation uncertainty.
- Fine-tune parameters: initial setting to estimate error covariance P, process noise Q, and measurement noise R.
- Instead of Python, code in C++ for performance computation.
  
