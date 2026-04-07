#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import sys
import time

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class PIDController:
    """A simple PID controller for smooth steering."""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0.0:
            dt = 0.001 # Prevent division by zero
            
        self.integral += error * dt
        # Anti-windup safety
        self.integral = max(-5.0, min(5.0, self.integral))
        
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        self.last_time = current_time
        return output

class linefollowing:
    def __init__(self):
        self.bridge = CvBridge()

        self.lost_line_counter = 0
        self.finish_triggered = False
        self.MAX_LOST_FRAMES = 10

        # Setup PID Controller 
        self.pid = PIDController(kp=20, ki=0.0, kd=0.0)

        # Set-up publisher subscriber relationship
        self.pub_cmd = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)
        self.image_sub = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.callback, queue_size=1)

        # Start timer for Fizz Detective
        rospy.loginfo("Initializing line follower... starting timer.")
        rospy.sleep(1)
        self.pub_score.publish("rover,123,0,aaaaaa")

    def callback(self, data):
        if self.finish_triggered:
            return
            
        cmd = Twist()

        # Attempts to convert image to cv2
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        height, width, _ = cv_image.shape
        
        # 1. Define FOV to save processing (Crop bottom 40%)
        fov = cv_image[int(height * 0.6) : height, :]
        fov_height, fov_width, _ = fov.shape
        
        # 2. Convert to HSV and threshold for the road color
        blurred = cv2.medianBlur(fov, 5)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 200])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # Clean up noise 
        kernel = np.ones((5,5), np.uint8)
        img_final = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        img_final = cv2.morphologyEx(img_final, cv2.MORPH_CLOSE, kernel)

        # 3. Find Max Contours
        contours, _ = cv2.findContours(img_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- RECOVERY LOGIC ---
        if len(contours) == 0:
            self.lost_line_counter += 1
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5 # Spin to find road
            self.pub_cmd.publish(cmd)
            
            cv2.imshow("Hybrid Vision", fov)
            cv2.waitKey(1)
            return
            
        else:
            self.lost_line_counter = 0
            
            # --- VISION LOGIC ---
            largest = max(contours, key=cv2.contourArea)
            final_road = np.zeros_like(img_final)
            cv2.drawContours(final_road, [largest], -1, 255, thickness=cv2.FILLED)

            # Find dynamic highest scanline (Look-ahead)
            x, y, w, h = cv2.boundingRect(largest)
            # Drop down 10% from the top of the road to get a solid slice
            scanline_row = int(min(y + h * 0.10, fov_height - 1))
            
            row_pixels = final_road[scanline_row, :]
            road_indices = np.where(row_pixels == 255)[0]

            if len(road_indices) == 0:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5
                self.pub_cmd.publish(cmd)
                return

            left_edge = road_indices[0]
            right_edge = road_indices[-1]

            # --- GEOMETRIC CENTERING ("Comparing Levers") ---
            # By calculating the true midpoint between the two visible edges, 
            # we perfectly balance the left and right gap distances.
            midpoint = (left_edge + right_edge) / 2.0
            
            # The error is the distance from the camera center to this midpoint.
            raw_error = (fov_width / 2.0) - midpoint

            # Normalize error to [-1.0, 1.0]
            normalized_error = raw_error / (fov_width / 2.0)

            # --- PID CONTROL LOGIC ---
            cmd.linear.x = 0.6 # Forward speed
            
            # Steer using the PID output
            steering_output = self.pid.compute(normalized_error)
            
            # Clamp the steering to prevent sudden violent jerks
            cmd.angular.z = max(-1.5, min(1.5, steering_output))

            self.pub_cmd.publish(cmd)

            # --- DEBUG DISPLAY ---
            # Draw the target midpoint we are aiming for
            cv2.circle(fov, (int(midpoint), scanline_row), 5, (0, 0, 255), -1) 
            # Draw a line showing the scanline we used
            cv2.line(fov, (0, scanline_row), (fov_width, scanline_row), (0, 255, 0), 1)
            cv2.imshow("Hybrid Vision", fov)
            cv2.waitKey(1)

def main(args):
    rospy.init_node('timetrials_move', anonymous=True)
    lf = linefollowing()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)