import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline


calib_image_dir = 'camera_cal/'
list_of_calib_images= glob.glob(calib_image_dir+'*.jpg')


def do_camera_calibration(test=False):

    # Calibration pattern
    nx = 9
    ny = 6


    # Arrays to store object points and image points from all the images
    objpoints = []
    imgpoints = []

    # prepare object points
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    #Iterate through all the images to find the image points
    for fname in list_of_calib_images:
        # Read the file
        #print(fname)
        img = cv2.imread(fname)
        #plt.imshow(img)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add corners/imgpoints
        if ret == True:
            #print(len(corners))
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #f, (ax1) = plt.subplots(1, 1, figsize=(24, 9))
            #f.tight_layout()
            #ax1.imshow(img)
            #ax1.set_title('Calibrating by finding imgpoints/corners in Image', fontsize=50)

    #Now lets do camera calibration (finding the camera matrix and distortion parameters)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2], None, None)

    if test is True:
        fname = list_of_calib_images[0]

        img = cv2.imread(fname)
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undistorted)
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    return mtx, dist




