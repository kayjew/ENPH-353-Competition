#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import sys
import time
import math

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

class PIDController:
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
        if dt <= 0.0: dt = 0.001 
        self.integral += error * dt
        self.integral = max(-5.0, min(5.0, self.integral))
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        self.last_time = current_time
        return output

class DirtRoadTester:
    def __init__(self):
        self.bridge = CvBridge()
        # MATCHED: Your highly responsive tuned PID values
        self.pid = PIDController(kp=40.0, ki=0.1, kd=0.2) 
        self.pub_cmd = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
        self.pub_score = rospy.Publisher('/score_tracker', String, queue_size=1)
        self.current_frame = None
        
        self.teleport_to_dirt_road()
        self.image_sub = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.callback, queue_size=1)

        rospy.loginfo("Initializing Pool-Proof Dirt Road Tester (Band-Pass Mode)...")
        rospy.sleep(1)
        self.pub_score.publish("rover,123,0,aaaaaa")

    def nothing(self, x): pass

    def teleport_to_dirt_road(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = 'B1' 
            state_msg.pose.position.x, state_msg.pose.position.y, state_msg.pose.position.z = -4.0, -2.25, 0.1 
            yaw = 0
            state_msg.pose.orientation.z, state_msg.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
            set_state(state_msg)
        except rospy.ServiceException as e: rospy.logerr(f"Teleport failed: {e}")

    def callback(self, data):
        try: self.current_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e: print(e)

    def run(self):
        cv2.namedWindow("Tuner Dashboard")
        # MATCHED: All your previous tuned values
        cv2.createTrackbar("Threshold", "Tuner Dashboard", 160, 255, self.nothing)
        cv2.createTrackbar("Median Kernel", "Tuner Dashboard", 10, 21, self.nothing)
        cv2.createTrackbar("Min Sat", "Tuner Dashboard", 20, 255, self.nothing)
        # NEW: Max Saturation Cap
        cv2.createTrackbar("Max Sat", "Tuner Dashboard", 125, 255, self.nothing)

        rate = rospy.Rate(15) 
        while not rospy.is_shutdown():
            if self.current_frame is not None:
                # 1. Downscale & Crop
                small = cv2.resize(self.current_frame, (0,0), fx=0.5, fy=0.5)
                h, w = small.shape[:2]
                fov = small[int(h*0.6):h, :]
                fov_h, fov_w = fov.shape[:2]
                
                # 2. Convert Colors
                gray = cv2.cvtColor(fov, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(fov, cv2.COLOR_BGR2HSV)
                
                # 3. MASK A: Dark Road (Grayscale INV)
                k = cv2.getTrackbarPos("Median Kernel", "Tuner Dashboard")
                if k % 2 == 0: k += 1
                blurred = cv2.medianBlur(gray, max(1, k))
                
                t = cv2.getTrackbarPos("Threshold", "Tuner Dashboard")
                _, road_mask = cv2.threshold(blurred, t, 255, cv2.THRESH_BINARY_INV) 

                # 4. MASK B: Saturation Band-Pass
                s_channel = hsv[:, :, 1]
                s_min = cv2.getTrackbarPos("Min Sat", "Tuner Dashboard")
                s_max = cv2.getTrackbarPos("Max Sat", "Tuner Dashboard")
                
                # We use inRange to get everything BETWEEN the two values
                s_mask = cv2.inRange(s_channel, s_min, s_max)

                # 5. COMBINE MASKS
                mask = cv2.bitwise_and(road_mask, s_mask)

                # Cleanup noise
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                # 6. LARGEST CONTOUR LOGIC
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                target_x = fov_w / 2.0
                road_found = False

                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_cnt) > 500:
                        M = cv2.moments(largest_cnt)
                        if M["m00"] > 0:
                            target_x = int(M["m10"] / M["m00"])
                            road_found = True
                            cv2.drawContours(fov, [largest_cnt], -1, (0, 255, 0), 2)

                # 7. PID & DRIVING
                cmd = Twist()
                if not road_found:
                    cmd.angular.z = 0.5 
                else:
                    error = (fov_w / 2.0) - target_x
                    normalized_error = error / (fov_w / 2.0)
                    cmd.linear.x = 0.4 
                    cmd.angular.z = self.pid.compute(normalized_error)
                self.pub_cmd.publish(cmd)

                # 8. DEBUG VIEW
                cv2.circle(fov, (int(target_x), int(fov_h/2)), 5, (0, 0, 255), -1) 
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                # Show current range on mask
                cv2.putText(mask_bgr, f"S-Range: {s_min}-{s_max}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Tuner Dashboard", np.hstack((fov, mask_bgr)))
                cv2.imshow("og view", self.current_frame)
            
            cv2.waitKey(5)
            rate.sleep()

def main(args):
    rospy.init_node('dirt_road_test', anonymous=True)
    tester = DirtRoadTester()
    try: tester.run()
    except KeyboardInterrupt: pass
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)