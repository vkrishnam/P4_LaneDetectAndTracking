import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from math import *
import collections
from itertools import chain
from functools import reduce
from scipy.signal import find_peaks_cwt
from moviepy.editor import VideoFileClip
import sys, getopt

import cameraCalib
#%matplotlib inline

cam_mtx = None
cam_dist = None
M = None
Minv = None

prints = False
frameCount = 0
left_lane = None
right_lane = None


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    # Calculate directional gradient
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    binary_output = np.copy(sbinary) # Remove this line

    # Return the binary image
    return binary_output



def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def color_thresh(img, thresh=(0, 255)):
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    binary_output =  np.zeros_like(img)
    binary_output[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary_output


def get_thresholded_image(img):
    # Gradient based
    abs_bin_threshold_image = abs_sobel_thresh(img, sobel_kernel=11, thresh=(50, 100))
    mag_bin_threshold_image = mag_thresh(img, sobel_kernel=21, thresh=(80, 100))
    dir_bin_threshold_image = dir_thresh(img, sobel_kernel=15, thresh=(0.7, 1.0))

    com_grad_threshold_bin_image = np.zeros_like(abs_bin_threshold_image).astype(np.uint8)
    com_grad_threshold_bin_image[(abs_bin_threshold_image == 1) | (mag_bin_threshold_image == 1) ] = 1
    # Color based
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls  = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    r = img[:,:,0]
    h = hls[:,:,0]
    l = hls[:,:,1]
    s = hls[:,:,2]

    gray_bin = color_thresh(gray, thresh=(200, 255))
    s_bin = color_thresh(s, thresh=(150, 255))
    r_bin = color_thresh(r, thresh=(200, 255))
    com_color_threshold_bin_image = np.zeros_like(abs_bin_threshold_image).astype(np.uint8)
    com_color_threshold_bin_image[(gray_bin == 1) | (s_bin == 1) | (r_bin == 1) ] = 1
    com_color_threshold_bin_image[(com_grad_threshold_bin_image==1)] = 1

    return com_color_threshold_bin_image

def getPerspectiveMatrices(img, src=None):
    global M, Minv

# | Source        | Destination   |
#|:-------------:|:-------------:|
#| 585, 460      | 320, 0        |
#| 203, 720      | 320, 720      |
#| 1127, 720     | 960, 720      |
#| 695, 460      | 960, 0        |


    img_size = (img.shape[1], img.shape[0])
    if src is None:
        src_vertices = np.array([[[585,455],[705, 455], [1130,720], [190, 720]]], dtype=np.int32)
        src = np.float32(src_vertices[0])
    #print(src)
    dst = np.float32(
        [
            [300,  100],
            [1000, 100],
            [1000, 720],
            [300,  720]
        ])
    #print(dst)
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)

    return M, Minv


def warp(img, src=None):
    global M, Minv
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)
    return warped



def find_perspective_points(edges):

    # Computing perspective points automatically
    rho = 2              # distance resolution in pixels of the Hough grid
    theta = 1*np.pi/180  # angular resolution in radians of the Hough grid
    threshold = 100       # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 100 # minimum number of pixels making up a line
    max_line_gap = 25    # maximum gap in pixels between connectable line segments

    angle_min_mag = 20*pi/180
    angle_max_mag = 65*pi/180

    lane_markers_x = [[], []]
    lane_markers_y = [[], []]

    masked_edges = np.copy(edges)
    masked_edges[:edges.shape[0]*5//10,:] = 0
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    for line in lines:
        for x1,y1,x2,y2 in line:
            theta = atan2(y1-y2, x2-x1)
            rho = ((x1+x2)*cos(theta) + (y1+y2)*sin(theta))/2
            if (abs(theta) >= angle_min_mag and abs(theta) <= angle_max_mag):
                if theta > 0: # positive theta is downward in image space?
                    i = 0 # Left lane marker
                else:
                    i = 1 # Right lane marker
                lane_markers_x[i].append(x1)
                lane_markers_x[i].append(x2)
                lane_markers_y[i].append(y1)
                lane_markers_y[i].append(y2)

    if len(lane_markers_x[0]) < 1 or len(lane_markers_x[1]) < 1:
        # Failed to find two lane markers
        return None

    p_left  = np.polyfit(lane_markers_y[0], lane_markers_x[0], 1)
    p_right = np.polyfit(lane_markers_y[1], lane_markers_x[1], 1)

    # Find intersection of the two lines
    apex_pt = np.linalg.solve([[p_left[0], -1], [p_right[0], -1]], [-p_left[1], -p_right[1]])
    top_y = ceil(apex_pt[0] + 0.0015*edges.shape[0])

    bl_pt = ceil(np.polyval(p_left, edges.shape[0]))
    tl_pt = ceil(np.polyval(p_left, top_y))

    br_pt = ceil(np.polyval(p_right, edges.shape[0]))
    tr_pt = ceil(np.polyval(p_right, top_y))

    src = np.array([[tl_pt, top_y],
                    [tr_pt, top_y],
                    [br_pt, edges.shape[0]],
                    [bl_pt, edges.shape[0]]], np.float32)

    return src




def findLanesSlidingWindow(warped_img):
    # Find the basePoints from Histogram
    #print(warped_img.shape)
    histogram = np.sum(warped_img[warped_img.shape[0]/2:,:], axis=0)
    #plt.plot(histogram)

    binary_warped = warped_img
    out_img = np.dstack((warped_img, warped_img, warped_img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint



    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []


    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


    return left_fit, right_fit, leftx, lefty, rightx, righty

def findLanesFromLaneFit(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, leftx, lefty, rightx, righty

def findCurvatureAndOffset(warped_image, left_fit, right_fit, leftx, lefty, rightx, righty):
    y_eval = warped_image.shape[0]
    #print('y_eval :', y_eval)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    if prints:
        print(left_curverad, 'm', right_curverad, 'm')

    #Lane/Road center in pixels
    left_fitx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_fitx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

    road_centre_in_pixels = (left_fitx+right_fitx)/2
    image_centre_in_pixels = warped_image.shape[1]/2

    diff_in_pixels = (image_centre_in_pixels-road_centre_in_pixels)
    diff_in_meters = diff_in_pixels*xm_per_pix

    if prints:
        print('Car center is offset from road centre (in m)', diff_in_meters)

    return left_curverad, right_curverad, diff_in_meters

def superImposeLanes(undistorted, binary_warped, left_fit, right_fit, left_r, right_r, dx, Minv, disabled=False):
    warped = binary_warped

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # DeWarp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted.shape[1], undistorted.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.4, 0)

    status = [ 'Engaged', 'Disengaged']
    index = int(disabled)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Left radius of curvature  = %.2f m'%(left_r),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,'Right radius of curvature = %.2f m'%(right_r),(50,80), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,'Vehicle position : %.2f m %s of center'%(abs(dx), 'left' if dx < 0 else 'right'),(50,110),
                        font, 1,(255,255,255),2,cv2.LINE_AA)
    #if index == 0:
    #    cv2.putText(result,'Tracking  = %s '%(status[index]),(50,140), font, 1,(0,255,0),2,cv2.LINE_AA)
    #else:
    #    cv2.putText(result,'Tracking  = %s '%(status[index]),(50,140), font, 1,(255,0,0),2,cv2.LINE_AA)


    return result

def findLanesInTestImages(folderName='test_images/'):
    mtx = cam_mtx
    dist = cam_dist
    global M, Minv
    list_of_test_images= glob.glob(folderName+'*.jpg')
    for imageFile in list_of_test_images:
        fname = imageFile
        print('Processing image...', fname)
        # Read the image
        img = mpimg.imread(fname)
        # Do Distortion Correction
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)

        # Find the combined thresholded image
        com_thresh_binary = get_thresholded_image(undistorted)
        # Optional: Find the perspective points
        src = find_perspective_points(com_thresh_binary)
        # Get Perspective matrices
        M, Minv = getPerspectiveMatrices(com_thresh_binary, src=None)
        # Do the Perspective transform
        warped_image= warp(com_thresh_binary)

        # Find lanes by Sliding Window
        left_fit, right_fit, leftx, lefty, rightx, righty = findLanesSlidingWindow(warped_image)
        # Find the curvature of Lanes and Road Offset
        left_curv_in_m, right_curv_in_m, vehicle_offset = findCurvatureAndOffset(warped_image, left_fit, right_fit, leftx, lefty, rightx, righty)
        # Super Impose the lanes on the image
        superImposed = superImposeLanes(undistorted, warped_image, left_fit, right_fit, left_curv_in_m, right_curv_in_m, vehicle_offset, Minv)
        # Save the Super Imposed Images
        filename, suffix = fname.split('.')
        filename = filename+'_lanesFound.jpg'
        mpimg.imsave(filename, superImposed)
    return

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.N = 10
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        self.cache_fit = []
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def update_fit(self, fit):
        self.current_fit = fit
        self.cache_fit.insert(0, fit)
        if len(self.cache_fit) > self.N:
            self.cache_fit.pop(-1)
        #print('cache',self.cache_fit)
        return

    def guidance_fit(self):

        if self.detected == False:
            return self.current_fit

        len_hist = len(self.cache_fit)
        if len_hist < 2:
            return self.current_fit

        rem_hist = len_hist - 1
        if rem_hist == 0:
            return self.current_fit
        else:
            other_weight = 1.0/rem_hist
            other_sum = np.sum(self.cache_fit[1:], axis=0)
            current_weight = 1.0
            current = self.cache_fit[0]
            fit = (0.50*other_weight*other_sum+1.50*current_weight*current)/2.0
            self.best_fit = fit
            #print('fit ', fit)
        return fit

    def updateAndCheckSanity(self, curve, other_curve, offset):
        if self.radius_of_curvature is None:
            self.radius_of_curvature = curve
            self.line_base_pos = offset
        else:

            if np.abs(curve-self.radius_of_curvature)/self.radius_of_curvature > 0.5:
                self.detected = False
                self.cache_fit = []
            else:
                self.detected = True
            self.radius_of_curvature = curve
            self.line_base_pos = offset
        return

def process_image(img):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # DONE: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    mtx = cam_mtx
    dist = cam_dist
    global M, Minv
    global frameCount
    global left_lane
    global right_lane

    # Do Distortion Correction
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    equalized = np.zeros_like(undistorted)

    # Do some histogram equalization
    for i in range(3):
        equalized[...,i] = cv2.equalizeHist(undistorted[...,i])

    # Find the combined thresholded image
    #com_thresh_binary = get_thresholded_image(equalized)
    com_thresh_binary = get_thresholded_image(undistorted)
    # Optional: To find the perspective points
    src = find_perspective_points(com_thresh_binary)
    # Get Perspective matrices
    # This also might be needed only for one time
    M, Minv = getPerspectiveMatrices(com_thresh_binary, src=None)
    # Do the Perspective transform
    warped_image= warp(com_thresh_binary)

    #print('frameCnt', frameCount, '\n')
    # Instantiate those Lane classes - one for left and another for right
    if (frameCount == 0):
        left_lane = Line()
        right_lane = Line()

    #if ((frameCount % 30)== 0) or (left_lane.detected == False) or (right_lane.detected == False):
    if (left_lane.detected == False) or (right_lane.detected == False):
        #print('pre',frameCount)
        # Find lanes by Sliding Window
        left_fit, right_fit, leftx, lefty, rightx, righty = findLanesSlidingWindow(warped_image)
    else:
        #print('post',frameCount)
        # Find lanes by recent lane fit
        left_fit, right_fit, leftx, lefty, rightx, righty = findLanesFromLaneFit(warped_image, left_lane.guidance_fit(), right_lane.guidance_fit())
        #left_fit, right_fit, leftx, lefty, rightx, righty = findLanesFromLaneFit(warped_image, left_lane.current_fit, right_lane.current_fit)

    left_lane.update_fit(left_fit)
    right_lane.update_fit(right_fit)

    left_fit = left_lane.guidance_fit()
    right_fit = right_lane.guidance_fit()

    #left_lane.current_fit = left_fit
    #right_lane.current_fit = right_fit


    frameCount = frameCount+1


    # Find the curvature of Lanes and Road Offset
    left_curv_in_m, right_curv_in_m, vehicle_offset = findCurvatureAndOffset(warped_image, left_fit, right_fit, leftx, lefty, rightx, righty)
    left_lane.updateAndCheckSanity(left_curv_in_m, right_curv_in_m, vehicle_offset)
    right_lane.updateAndCheckSanity(right_curv_in_m, left_curv_in_m, vehicle_offset)
    # Super Impose the lanes on the image
    superImposed = superImposeLanes(undistorted, warped_image, left_fit, right_fit, left_curv_in_m, right_curv_in_m, vehicle_offset, Minv, ((left_lane.detected == False) or (right_lane.detected == False)) )



    return superImposed


def getCameraParameters(cam_file):
    if cam_file is None:
        cam_file = 'camera_parameters.npz'
    #global cam_mtx, cam_dist
    if not os.path.isfile(cam_file):
        print('Calibrating camera ...')
        cam_mtx, cam_dist = cameraCalib.do_camera_calibration(test=False)
        # Save it to a file so that next time the parameters can be read from
        # a file, rather processing calibration images
        print('Saving camera data to ', cam_file)
        np.savez_compressed(cam_file, cam_mtx=cam_mtx, cam_dist=cam_dist)
        print('Cam matrix: ', cam_mtx)
        print('Distortion Parameters: ', cam_dist)
    else:
        print('Loading camera data from', cam_file)
        data = np.load(cam_file)
        cam_mtx = data['cam_mtx']
        cam_dist = data['cam_dist']
    return cam_mtx, cam_dist


def main(argv):
    camFile = 'camera_parameters.npz'
    inputVideoFile = None
    outputVideoFile = None
    global cam_dist, cam_mtx
    try:
        opts, args = getopt.getopt(argv,"hi:o:c:",["ifile=","ofile=","cfile="])
    except getopt.GetoptError:
        print('p4.py -i <inputVideoFile> -o <outputVideoFile>  -c <cameraFile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('p4.py -i <inputVideoFile> -o <outputVideoFile>  -c <cameraFile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputVideoFile = arg
        elif opt in ("-o", "--ofile"):
            outputVideoFile = arg
        elif opt in ("-c", "--cfile"):
            camFile = arg
    print( 'Input Video file is "', inputVideoFile)
    print( 'Output Video file is "', outputVideoFile)
    print( 'Camera file is "', camFile)

    cam_mtx, cam_dist = getCameraParameters(camFile)
    if (inputVideoFile is None) | (outputVideoFile is None):
        findLanesInTestImages()
        sys.exit(0)

    #Do the video processing
    outfile = outputVideoFile
    clip1 = VideoFileClip(inputVideoFile)
    out_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    out_clip.write_videofile(outfile, audio=False)
    sys.exit(0)


if __name__ == "__main__":
   main(sys.argv[1:])

