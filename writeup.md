##Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./solution/undistort_output.png "Undistorted"
[image2]: ./test_images/test2.jpg "Road Transformed"
[image3]: ./solution/binary_combo_example.png "Binary Example"
[image4]: ./solution/warped_straight_lines.png "Warp Example"
[image5]: ./solution/color_fit_lines.png "Fit Visual"
[image6]: ./solution/example_output.png "Output"
[image7]: ./solution/hls_vs_hsv.png "Output"
[image8]: ./solution/hls_vs_hsv_2.png "Output"
[video1]: ./project.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 3rd code cell of the IPython notebook located in "./Solution.ipynb" and file `detector.py`.

The `detector.py` contains a `Camera` class which has a method called `calibrate`, accepting a list of file names and size of the points for calibration.   

In `calibrate` method, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I saved the matrix and coefficients as `Camera`'s properties. I also wrapped the OpenCV function  `cv2.undistort()` into `Camera` class to avoid the complex interface of OpenCV function.

In the 3rd cell, I called `camera.undistort()` to get the undistorted image. One of the example list as follows.

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color (on V layer of HSV channels), x Sobel and y Sobel thresholds to generate a binary image (thresholding steps at lines #123 through #152 in `detector.py`). Additionally, I also added a mask that will get rid of the noise that is out of the interest region, which is very similar in Project 1.

Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In the code of `Camera` class in `detector.py`, I define a method called `calculate_perspective_transform`, which accept two list of points, one is the source, the other is the destination. In the `calculate_perspective_transform` method, I use `cv2.getPerspectiveTransform(src, dst)` to calculate the transform matrix and then calculate the reversed matrix. The points are hardcoded as follows (you can see the code in 5th code cell):

```
src = np.array([(262, 682), (575, 464), (707, 464), (1049, 682)], dtype=np.float32)
dst = np.array([(300, 682), (300, 50), (980, 50), (980, 682)], dtype=np.float32)

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 262, 682     | 300, 682        | 
| 707, 464      | 300, 20      |
| 709, 465     | 980, 20      |
| 1049, 682      | 980, 682        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

In the image above, the left one is the original image, the middle one is the bird view of the original image, and the right one is marked with red on the left line and blue on the right line. Also, there are green lines that represent the fitted 2nd order polynomial.

The code that mark the lines are at #268 through #316 in `detector.py` method `detect`. I used convolution to find the center of the lines, which can be found in #148 throught #199 in `detector.py` function `find_line_center` and `find_window_centroids`.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #246 through #250 in my code in `Line` class in `detector.py` with method `append`. I decided to use `Line` class to remember recent 5 frames and every time a new frame is detected, I will calculate the curvature of the lane.

The offset calculation is done in the `LaneDetector` in `detetor.py` at line #367.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #349 through #365 in my code in `detector.py` in the function `mark_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The biggest challenge in this project was the shadow of trees in the image. When there is shadow, the threshold binary image will be much worse than without shadow. After trying for some time, I found that the V channel of HSV presentation of a image performs better when there is shadow. The comparision of HLS and HSV presentation of a image is as below:

![alt text][image7]

But V channel performs worse when the image is very bright. The road and the line are all very light in the image. As below:

![alt text][image8]

Actually, my pipeline will fail when the car went into some places that are very bright. If I have more time, I would add some condition logic that will use different channel in different situation. In the shadow condition, use V channel of HSV image; In the bright condition, use S channel of HLS image. But I do not think this is the final solution. I believe that machine learning could have some more robust solution. Actually, I think we can use deep learning to find lane points from the image.

To avoid some mad thing happened, I added sanity check in the detect pipeline in #386 in `detector.py`. As a result, there would be some frames without lane mark. I think in the real world, this may cause some problems.