from typing import List, Tuple, Dict

import numpy as np

from collections import deque

import cv2

ym_per_pix = 30 / 720
xm_per_pix = 3.7 / 700


class Camera(object):
    def __init__(self):
        self.matrix = None
        self.coefficient = None
        self.perspective_transform_matrix = None
        self.perspective_reverse_transform_matrix = None

    def undistort(self, img):
        return cv2.undistort(img, self.matrix, self.coefficient)

    def calibrate(self, file_paths: List[str], size: Tuple[int] = (6, 9)):
        assert len(size) == 2
        rows, cols = size
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

        obj_points = []
        img_points = []

        image_size = None

        for path in file_paths:
            img = cv2.imread(path)

            if image_size is None:
                image_size = img.shape[-2::-1]

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            is_found, corners = cv2.findChessboardCorners(gray, size[::-1])
            if is_found:
                obj_points.append(objp)
                img_points.append(corners)
        _, self.matrix, self.coefficient, _, _ = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None)

    def calculate_perspective_transform(self, src, dst):
        self.perspective_transform_matrix = cv2.getPerspectiveTransform(src, dst)
        self.perspective_reverse_transform_matrix = cv2.getPerspectiveTransform(dst, src)

    def warp_perspective(self, binary_image):
        size = None
        if len(binary_image.shape) == 3:
            size = binary_image.shape[-2::-1]
        elif len(binary_image.shape) == 2:
            size = binary_image.shape[::-1]
        return cv2.warpPerspective(
            binary_image,
            self.perspective_transform_matrix,
            size,
            flags=cv2.INTER_LINEAR
        )

    def unwarp_perspective(self, image):
        size = None
        if len(image.shape) == 3:
            size = image.shape[-2::-1]
        elif len(image.shape) == 2:
            size = image.shape[::-1]
        return cv2.warpPerspective(
            image,
            self.perspective_reverse_transform_matrix,
            size,
            flags=cv2.INTER_LINEAR
        )


def _scale_to_255(array):
    return np.uint(255 * array / np.max(array))


ImageWrapperParams = Dict[str, Dict[str, float]]


class ImageWrapper(object):
    def __init__(self, image, params: ImageWrapperParams):
        self.image = image

        self._grad_x_param = {'ksize': 3, 'min': 0., 'max': 255.}
        self._grad_x_param.update(params.get('grad_x', {}))
        self._grad_y_param = {'ksize': 3, 'min': 0., 'max': 255.}
        self._grad_y_param.update(params.get('grad_y', {}))
        self._grad_mag_param = {'min': 0., 'max': 255.}
        self._grad_mag_param.update(params.get('grad_mag', {}))
        self._grad_dir_param = {'min': 0., 'max': np.math.pi / 2}
        self._grad_dir_param.update(params.get('grad_dir', {}))
        self._grad_color_param = {'min': 0., 'max': 255.}
        self._grad_color_param.update(params.get('grad_color', {}))

        self.grad_x = None
        self.grad_y = None
        self.grad_mag = None
        self.grad_dir = None
        self.grad_color = None
        self.combined = None

        self.grad_x_bin = None
        self.grad_y_bin = None
        self.grad_mag_bin = None
        self.grad_dir_bin = None
        self.grad_color_bin = None

        self.generate()

    def _gen_bin(self, name: str):
        field = self.__dict__[name]
        param = self.__dict__["_{}_param".format(name)]
        res_bin = np.zeros_like(field)
        res_bin[(field > param["min"]) & (field <= param["max"])] = 1
        return res_bin

    def generate(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        grad_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self._grad_x_param['ksize']))
        grad_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self._grad_x_param['ksize']))

        self.grad_color = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)[:, :, 2]
        self.grad_color_bin = self._gen_bin("grad_color")
        # self.grad_mag = _scale_to_255(np.sqrt(grad_x ** 2 + grad_y ** 2))
        # self.grad_mag_bin = self._gen_bin("grad_mag")
        self.grad_x = _scale_to_255(grad_x)
        self.grad_x_bin = self._gen_bin("grad_x")
        self.grad_y = _scale_to_255(grad_y)
        self.grad_y_bin = self._gen_bin("grad_y")
        # self.grad_dir = np.arctan2(grad_y, grad_x)
        # self.grad_dir_bin = self._gen_bin("grad_dir")

        self.combined = np.zeros_like(gray)
        self.combined[
            ((self.grad_x_bin == 1) & (self.grad_y_bin == 1)) |
            # ((self.grad_mag_bin == 1) & (self.grad_dir_bin == 1)) |
            (self.grad_color_bin == 1)
            ] = 255

        height, width = gray.shape
        roi_height = 440
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [np.array(
            [(0, int(height)), (int(width / 2 * 0.9), roi_height),
             (int(width / 2 * 1.1), roi_height), (width, int(height))])], 255)
        self.combined = cv2.bitwise_and(self.combined, mask)


def find_line_center(binary_image, window_width):
    window = np.ones(window_width)
    summed = np.sum(binary_image, axis=0)
    center = np.argmax(np.convolve(window, summed)) - window_width / 2
    return center


# TODO: refactor
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


class Line(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=5)
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = 0.
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def append(self, line_points):
        self.ally, self.allx = line_points.nonzero()
        # if ally is 0-size, it means that there are no point detected as lane.
        self.current_fit = np.polyfit(self.ally, self.allx, 2)

        ploty = np.linspace(0, line_points.shape[0] - 1, line_points.shape[0])
        fitx = self.current_fit[0] * ploty ** 2 + self.current_fit[1] * ploty + self.current_fit[2]
        self.recent_xfitted.append(fitx)

        self.bestx = np.average(np.vstack(self.recent_xfitted), axis=0)

        y_eval = np.max(self.ally)
        left_fit_cr = np.polyfit(self.ally * ym_per_pix, self.allx * xm_per_pix, 2)
        self.radius_of_curvature = \
            0.1 * self.radius_of_curvature + \
            0.9 * ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / \
            np.absolute(2 * left_fit_cr[0])


class LaneDetector(object):
    def __init__(self, camera: Camera, wrapper_params: ImageWrapperParams):
        self.camera = camera
        self.wrapper_params = wrapper_params

        self.window_width = 50
        self.window_height = 40  # Break image into 9 vertical layers since image height is 720
        self.margin = 100  # How much to slide left and right for searching

        self.warped = None
        self.mask = None
        self.image_with_mask = None

        self.left_line = Line()
        self.right_line = Line()

        self.offset = None
        self.colored_warp_image = None

    def detect(self, image):
        image = self.camera.undistort(image)
        wrapper = ImageWrapper(image, self.wrapper_params)
        self.warped = self.camera.warp_perspective(wrapper.combined)

        window_centroids = self.find_window_centroids(self.warped)

        # Points used to draw all the left and right windows
        l_points, r_points = self._find_window_points(self.warped, window_centroids)

        self.draw_window_mask(l_points, r_points, self.warped)

        left_points, right_points = self._find_line_points(l_points, r_points, self.warped)

        if self.sanity_check():
            self.offset = self.get_offset()
            self._draw_colored_lines(left_points, right_points, self.warped)
            result = self.mark_lane(image, left_points, right_points)
            self.add_text_to_result(result)
            return result
        else:
            return image

    def find_window_centroids(self, warped):
        window_centroids = []  # Store the (left,right) window centroid positions per level
        window = np.ones(self.window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by
        # using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        # l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        # l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        # r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        # r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)
        left_bottom_image = warped[int(2 * warped.shape[0] / 3):, :int(warped.shape[1] / 2)]
        right_bottom_image = warped[int(2 * warped.shape[0] / 3):, int(warped.shape[1] / 2):]
        l_center = find_line_center(left_bottom_image, self.window_width)
        r_center = find_line_center(right_bottom_image, self.window_width) + int(warped.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(warped.shape[0] / self.window_height)):
            # convolve the window into the vertical slice of the image
            layer_bottom = int(warped.shape[0] - (level + 1) * self.window_height)
            layer_top = int(warped.shape[0] - level * self.window_height)
            image_layer = np.sum(warped[layer_bottom:layer_top, :], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is
            # at right side of window, not center of window
            offset = self.window_width / 2
            l_min_index = int(max(l_center + offset - self.margin, 0))
            l_max_index = int(min(l_center + offset + self.margin, warped.shape[1]))
            if np.max(conv_signal[l_min_index:l_max_index]) > 100:
                l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - self.margin, 0))
            r_max_index = int(min(r_center + offset + self.margin, warped.shape[1]))
            if np.max(conv_signal[r_min_index:r_max_index]) > 100:
                r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # Add what we found for that layer
            window_centroids.append((l_center, r_center))

        return window_centroids

    def _find_line_points(self, l_points, r_points, warped):
        left_points = np.zeros_like(warped)
        left_points[(warped > 128) & (l_points == 255)] = 1
        right_points = np.zeros_like(warped)
        right_points[(warped > 128) & (r_points == 255)] = 1
        self.left_line.append(left_points)
        self.right_line.append(right_points)
        return left_points, right_points

    def _draw_colored_lines(self, left_points, right_points, warped):
        warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
        self.colored_warp_image = warpage
        self.colored_warp_image[left_points.nonzero()] = [255, 0, 0]
        self.colored_warp_image[right_points.nonzero()] = [0, 0, 255]

    def draw_window_mask(self, l_points, r_points, warped):
        # add both left and right window pixels together
        template = np.array(r_points + l_points, np.uint8)
        # create a zero color channle
        zero_channel = np.zeros_like(template)
        # make window pixels green
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
        # making the original road pixels 3 color channels
        warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
        # overlay the orignal road image with window results
        self.image_with_mask = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

    def _find_window_points(self, warped, window_centroids):
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(self.window_width, self.window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(self.window_width, self.window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        return l_points, r_points

    def add_text_to_result(self, result):
        curvature = (self.left_line.radius_of_curvature + self.right_line.radius_of_curvature) / 2
        cv2.putText(
            result,
            "Radius of Curvature = {:.2f}(m)".format(curvature.item()),
            (100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        offset_value = np.absolute(self.offset).item()
        offset_direction = "right" if self.offset < 0 else "left"
        cv2.putText(
            result,
            "Vehicle is {:.2f}m {} of the road".format(offset_value, offset_direction),
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    def mark_lane(self, image, left_points, right_points):
        color_warp = np.zeros_like(image)
        ploty = np.linspace(0, self.image_with_mask.shape[0] - 1,
                            self.image_with_mask.shape[0])
        left_fitx = self.left_line.bestx
        right_fitx = self.right_line.bestx
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        newwarp = cv2.warpPerspective(color_warp, self.camera.perspective_reverse_transform_matrix,
                                      (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        result[self.camera.unwarp_perspective(left_points).nonzero()] = [255, 0, 0]
        result[self.camera.unwarp_perspective(right_points).nonzero()] = [0, 0, 255]
        return result

    def get_offset(self):
        return ((self.left_line.bestx[-1] + self.right_line.bestx[-1]) / 2 - 640) * xm_per_pix

    def sanity_check(self):
        line_distance = self.right_line.bestx - self.left_line.bestx
        return 650 < np.average(line_distance) < 750 and \
               np.std(line_distance) < 50
