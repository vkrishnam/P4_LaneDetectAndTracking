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
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

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

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
