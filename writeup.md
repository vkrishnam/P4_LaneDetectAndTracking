##Project Writeup 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* [x] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* [x] Apply a distortion correction to raw images.
* [x] Use color transforms, gradients, etc., to create a thresholded binary image.
* [x] Apply a perspective transform to rectify binary image ("birds-eye view").
* [x] Detect lane pixels and fit to find the lane boundary.
* [x] Determine the curvature of the lane and vehicle position with respect to center.
* [x] Warp the detected lane boundaries back onto the original image.
* [x] Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/orig_undistorted_calibration.png "Original-Undistorted"
[image2]: ./output_images/test_undistorted.png "Test-Undistorted"
[image3]: ./output_images/Test_FinalThresholdBinaryImage.png "Binary Example"
[image5]: ./output_images/Test_dst.png "Test Dst area"
[image4]: ./output_images/Test_src.png "Test Src area"
[image6]: ./output_images/SlidingWindow.png "Sliding Window"
[image7]: ./output_images/MaskingApproach.png "Masking"
[image8]: ./output_images/LanesFound.png "LanesFound"
[video1]: ./project_video.mp4 "Video"
[image9]: ./output_images/Gray_Channel.png "Gray"
[image10]: ./output_images/H_Channel.png "Hue"
[image11]: ./output_images/R_Channel.png "Red"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  
Here is one and You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in cameraCalib.py.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners (9x6) in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

Also I write the Camera Calibration parameters of Camere Matrix and Distortion parameters to a file so that it can be looked up in later iterations.

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I analyzed the various color channels and gradient information such as absolute value, magnitude and direction. I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #100 through #121 in `p4.py`).  Here's an example of my output for this step. 

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes functions called `warp()` and `getPerspectiveMatrices()`, in the file `p4.py`.  The `getPerspectiveMatrices()` function computes the Perspective matrices with the src and dest mapping given. I have chose to hardcode the source and destination points after experimenting in the Jupyter notebook. The `warp()` function takes as inputs an image (`img`), and returns the warped Image.  I chose the hardcode the source and destination points in the following manner:

```
    src_vertices = np.array([[[585,455],[705, 455], [1130,720], [190, 720]]], dtype=np.int32)
    src = np.float32(src_vertices[0])
    dst = np.float32(
        [
            [300,  100],
            [1000, 100],
            [1000, 720],
            [300,  720]
        ])

```
This resulted in the following source and destination points:

| Source         | Destination   | 
|:--------------:|:-------------:| 
| 585,  460      | 320, 0        | 
| 203,  720      | 320, 720      |
| 1127, 720      | 960, 720      |
| 695,  460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the warped ("top-view") image, the Histogram and sliding window is carried to identify the lane-line pixel and whereafter `cv2.polyfit()` is used to find a second order polynomial to fit lane like this:

![alt text][image6]

This is carried out by the function `findLanesSlidingWindow()` in p4.py

When the lane fit is already available, say from previous frame, we go about doing a masking with some margin in order to avoid compute intensive sliding window approach, whose result can be visualized like shown:

![alt text][image7]

This is carried out by the function `findLanesFromLaneFit()` in p4.py

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This computation is in the function `findCurvatureAndOffset` in p4.py file.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This step is implemented by the function `superImposeLanes()` in p4.py.  Here is an example of my result on a test image:

![alt text][image8]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

Basic building blocks of the pipeline are:
* Distortion Correction
* Finding the thresholded binary image
* Perspective transform
* Lane detection and tracking
* Lane curvature and Vehicle offset calculation 
* Super imposition of the lane onto Undistorted image frame

##### Distortion Correction
Once the Camera calibration is done successfully, which gives Camera Matrix and Distortion Parameters, Distortion correction is straight forward step. Also this step is invariant to the different scenario videos as the camera instrinsic do not change.

##### Finding the thresolded binary image.
This is one of the challenging step, as we cannot find line pixels very predictively from one single source or channel. Hence we consider many color channels in different domains and gradient based methods. Even here the issue is the threshold values which might work for a scene might not work for different scene given the differences coming from illumination, shadows, road texture. Also it is being explored if histogram equalization helps in those scenarios of extreme contrast. 

![alt text][image9]
![alt text][image10]
![alt text][image11]

##### Perspective transform
Here what is learned that the perspective transform arrived and which successfully for one video sequence does not old true for the more challenging video sequence becuase of the orientation of the vehicle with respective to road. Hence even the Perspective transform matrices have to be calculated newly for new video and also would be good to do it online/adaptively as the scene changes. Another irony is if we want to larger distance from the vehicle covered which is good for more accurate road
curvature calculation, then we might suffer on the accurate lane detection part.


##### Lane detection and tracking
Here the fact that the lanes does not change direction so drastically in subsequent frames is exploited where a history lanes detected in previous 10 frames is cache which gives guidance as an approximate lane fit where the lanes are searched for given a margin of +/-100 pixels. In case of failure to find lanes which would be shown as drastic diff in road curvature, the fall back  but a slightly compute intensive method of SlidingWindow is pursued.

##### Lane curvature and Vehicle offset calculation 
If the lanes detected are right, this is a straight forward step. Here the hyperparameters used are.
```
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

##### Super imposition of the lane onto Undistorted image frame
Given the line fit of left and right lanes, the area is filled with a color and dewarped on to the undistorted image domain where it is additively added to shown transparency effect.



#### Future work
To make the steps of  
* Finding the thresolded binary image.
* Perspective transform
more robust to make it work with the harder challenge and more real world scenarios.



