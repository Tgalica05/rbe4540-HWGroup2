"""Simple example: receive camera images and send poses and velocities. 
Display masked palm camera image."""

from asyncio import wait
import os
import time
from datetime import datetime

from matplotlib import image

import rclpy

import numpy as np
import cv2

from common_interfaces_merlab.srv import SendPose, SendTwist
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from matplotlib import pyplot as plt


class SimpleRun(Node):
    def __init__(self):
        super().__init__('simple_run')
        self.cartesian_client = self.create_client(SendPose, '/cartesian_ref')
        self.ee_velocity_client = self.create_client(SendTwist, '/set_ee_velocity')
        self.bridge = CvBridge()
        self.palm_image = None  # Latest OpenCV image; None until one arrives.
        self.palm_camera_subscriber = self.create_subscription(
            Image,
            '/palm_camera/image',
            self.palm_image_callback,
            qos_profile_sensor_data,
        )

    def palm_image_callback(self, msg):
        """Convert the ROS image to an OpenCV image (BGR NumPy array)."""
        self.palm_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Add your image processing here.

        # use np array to store color RGB color bounds array[B, G, R]   
        # RGB bounds (AI generated):
        yellow_lower = np.array([0, 100, 100])
        yellow_upper = np.array([50, 255, 255])
        green_lower = np.array([0, 100, 0])
        green_upper = np.array([50, 255, 50])
        blue_lower = np.array([210, 0, 0])
        blue_upper = np.array([255, 30, 30])
        cyan_lower = np.array([210, 210, 0])
        cyan_upper = np.array([255, 255, 30])


        # create binary masks of the pixels that are in range
        mask_bin_yellow = cv2.inRange(self.palm_image, yellow_lower, yellow_upper)
        mask_bin_green = cv2.inRange(self.palm_image, green_lower, green_upper)
        mask_bin_blue = cv2.inRange(self.palm_image, blue_lower, blue_upper)
        mask_bin_cyan = cv2.inRange(self.palm_image, cyan_lower, cyan_upper)

        # combine all colors to create mask (for display)
        mask_bin = mask_bin_yellow + mask_bin_green + mask_bin_blue + mask_bin_cyan

        # Find the centroids of the remaining pixels for each color mask
        M_y = cv2.moments(mask_bin_yellow)
        M_g = cv2.moments(mask_bin_green)
        M_b = cv2.moments(mask_bin_blue)
        M_c = cv2.moments(mask_bin_cyan)

        if M_y['m00'] != 0:
            Cx_y = int(M_y['m10'] / M_y['m00'])
            Cy_y = int(M_y['m01'] / M_y['m00'])
        else:
            Cx_y, Cy_y = 0, 0

        if M_g['m00'] != 0:
            Cx_g = int(M_g['m10'] / M_g['m00'])
            Cy_g = int(M_g['m01'] / M_g['m00'])
        else:
            Cx_g, Cy_g = 0, 0

        if M_b['m00'] != 0:
            Cx_b = int(M_b['m10'] / M_b['m00'])
            Cy_b = int(M_b['m01'] / M_b['m00'])
        else:
            Cx_b, Cy_b = 0, 0

        if M_c['m00'] != 0:
            Cx_c = int(M_c['m10'] / M_c['m00'])
            Cy_c = int(M_c['m01'] / M_c['m00'])
        else:
            Cx_c, Cy_c = 0, 0

        # create a colored mask to show the remaining pixels in color
        mask_bgr = cv2.bitwise_and(self.palm_image, self.palm_image, mask=mask_bin)

        # Detect edges on binary mask
        # image_edges = cv2.Canny(mask_bin, 100, 200)

        # override the original image with the mask
        self.palm_image = mask_bgr

        # draw centroids on img
        # we do this before we override pix coordinates with camera coordinates
        cv2.circle(self.palm_image, (int(Cx_y), int(Cy_y)), 3, (255, 255, 255), -1)
        cv2.circle(self.palm_image, (int(Cx_g), int(Cy_g)), 3, (255, 255, 255), -1)
        cv2.circle(self.palm_image, (int(Cx_b), int(Cy_b)), 3, (255, 255, 255), -1)
        cv2.circle(self.palm_image, (int(Cx_c), int(Cy_c)), 3, (255, 255, 255), -1)

        # transform from image frame coordiantes to 
        Cx_y, Cy_y = self.img_to_cam(Cx_y, Cy_y)
        Cx_y = round(Cx_y, 3)
        Cy_y = round(Cy_y, 3)

        Cx_g, Cy_g = self.img_to_cam(Cx_g, Cy_g)
        Cx_g = round(Cx_g, 3)
        Cy_g = round(Cy_g, 3)

        Cx_b, Cy_b = self.img_to_cam(Cx_b, Cy_b)
        Cx_b = round(Cx_b, 3)
        Cy_b = round(Cy_b, 3)

        Cx_c, Cy_c = self.img_to_cam(Cx_c, Cy_c)
        Cx_c = round(Cx_c, 3)
        Cy_c = round(Cy_c, 3)

        # Print centroid locations on image
        cv2.putText(self.palm_image, f'Yellow: ({Cx_y} {Cy_y})', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(self.palm_image, f'Green: ({Cx_g} {Cy_g})', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(self.palm_image, f'Blue: ({Cx_b} {Cy_b})', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(self.palm_image, f'Cyan: ({Cx_c} {Cy_c})', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # display image
        cv2.imshow('Palm Camera', self.palm_image)
        cv2.waitKey(1) # refresh rate in milliseconds


    def img_to_cam(self, x_im, y_im):

        # define camera parameters
        s_x = 0.00001
        s_y = 0.00001
        o_x = 320
        o_y = 240
        f = 0.0032
        Z = 0.5

        # translate from pixel frame to image plane frame
        x = -(x_im - o_x)*s_x
        y = -(y_im - o_y)*s_y

        # translate from image plane frame to camera frame (in meters)
        X = (x*Z)/f
        Y = (y*Z)/f

        return X, Y


    def move_cartesian(self, x, y, z):
        """Move tool0 to a position in base_link (meters), pointing downward."""
        request = SendPose.Request()
        request.pose.position.x = float(x)
        request.pose.position.y = float(y)
        request.pose.position.z = float(z)
        # Quaternion (x, y, z, w) = (1, 0, 0, 0).
        request.pose.orientation.x = 1.0
        request.pose.orientation.w = 0.0
        request.pose.orientation.y = 0.0
        request.pose.orientation.z = 0.0

        self.get_logger().info(f'Moving to ({x:.2f}, {y:.2f}, {z:.2f})')
        future = self.cartesian_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=120.0)
        if not future.done():
            self.get_logger().error('No motion response; stopping the motion sequence')
            return False
        response = future.result()
        if response is None or not response.success:
            self.get_logger().error('Cartesian motion failed; stopping the motion sequence')
            return False
        return True

    def set_ee_velocity(self, vx=0.0, vy=0.0, vz=0.0,
                        wx=0.0, wy=0.0, wz=0.0):
        """Set tool-frame linear (m/s) and angular (rad/s) velocities."""
        request = SendTwist.Request()
        request.twist.linear.x = float(vx)
        request.twist.linear.y = float(vy)
        request.twist.linear.z = float(vz)
        request.twist.angular.x = float(wx)
        request.twist.angular.y = float(wy)
        request.twist.angular.z = float(wz)
        future = self.ee_velocity_client.call_async(request)
        # The first command also starts Servo through the motion interface.
        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
        if not future.done():
            self.get_logger().error('No velocity response; stopping the motion sequence')
            return False
        response = future.result()
        if response is None or not response.success:
            message = response.message if response is not None else 'No response'
            self.get_logger().error(f'Velocity command failed: {message}')
            return False
        return True

    def move_ee_velocity(self, vx=0.0, vy=0.0, vz=0.0,
                         wx=0.0, wy=0.0, wz=0.0, duration=1.0):
        """Refresh a tool-frame velocity for duration seconds; caller stops it."""
        self.get_logger().info(
            f'Tool-frame velocity: linear=({vx}, {vy}, {vz}) m/s, '
            f'angular=({wx}, {wy}, {wz}) rad/s for {duration:.1f} s'
        )
        if not self.set_ee_velocity(vx, vy, vz, wx, wy, wz):
            return False
        end_time = time.monotonic() + duration
        while rclpy.ok():
            remaining = end_time - time.monotonic()
            if remaining <= 0.0:
                return True
            # Refresh before the interface's default 0.5-second watchdog expires.
            time.sleep(min(0.1, remaining))
            if time.monotonic() >= end_time:
                return True
            if not self.set_ee_velocity(vx, vy, vz, wx, wy, wz):
                return False
        return False

    def run(self):
        self.get_logger().info('Waiting for /cartesian_ref...')
        while rclpy.ok():
            if self.cartesian_client.wait_for_service(timeout_sec=1.0):
                break
        if not rclpy.ok():
            return

        # TODO: Implement your homework here. Edit or extend these two moves.
        # Each call waits for the robot to finish before continuing.
        # Images are received while the motion methods spin waiting for replies.
        # To receive images outside those methods, call rclpy.spin_once(self).

        # move to grasp-ready pose
        if not self.move_cartesian(0.45, 0.0, 0.5):
            return
        # return to a different pose
        if not self.move_cartesian(0.60, 0.10, 0.5):

            return
        time.sleep(2.0)

        # AI generated chunk to timestamp the filename to prevent overriding previous image
        output_dir = '/home/tim-galica/RBE4540/Tim_Galica - HW5/OpenCV images'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_dir, f'image_{timestamp}.jpg')


        cv2.imwrite(filename, self.palm_image)



        self.get_logger().info('Waiting for /set_ee_velocity...')
        while rclpy.ok():
            if self.ee_velocity_client.wait_for_service(timeout_sec=1.0):
                break
        if not rclpy.ok():
            return

        # Move along tool0's +X, then -X (about 9 cm each at 0.03 m/s).
        # Switch directly between velocities; send zero when the sequence ends.
        # try:
        #   if not self.move_ee_velocity(vx=0.03, duration=3.0):
        #       return
        #   if not self.move_ee_velocity(vx=-0.03, duration=3.0):
        #       return
        #finally:
            # Also request a stop if a command fails or the user interrupts.
            # If ROS has shut down, the interface watchdog stops stale commands.
        #    stopped = self.set_ee_velocity() if rclpy.ok() else False
        # if not stopped:
        #    return
        self.get_logger().info('Motion sequence complete')


def main(args=None):
    rclpy.init(args=args)
    node = SimpleRun()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
