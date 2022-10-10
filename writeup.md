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
<video src="/results/my_tracking_results.avi">



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
  