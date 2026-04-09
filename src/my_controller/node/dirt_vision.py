#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge, CvBridgeError

class DirtVisionNode:
    def __init__(self):
        rospy.init_node('dirt_vision', anonymous=True)
        self.bridge = CvBridge()
        self.current_frame = None
        self.prev_frame = None # NEW: Store previous frame
        
        
        # Output Topics
        self.stuck_pub = rospy.Publisher('/vision/dirt_stuck', Bool, queue_size=1)
        self.error_pub = rospy.Publisher('/vision/dirt_error', Float32, queue_size=1)
        self.visible_pub = rospy.Publisher('/vision/dirt_visible', Bool, queue_size=1)
        
        self.image_sub = rospy.Subscriber('/B1/rrbot/camera1/image_raw', Image, self.callback, queue_size=1)
        rospy.loginfo("Dirt Vision Node Initialized")

    def callback(self, data):
        try: 
            self.current_frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e: 
            print(e)

    def run(self):
        cv2.namedWindow("Dirt Vision")

        # HARDCODED TUNED VALUES
        THRESH_VAL = 170
        K_SIZE = 11
        S_MIN = 20
        S_MAX = 255

        rate = rospy.Rate(30) 
        while not rospy.is_shutdown():
            if self.current_frame is not None:
                small = cv2.resize(self.current_frame, (0,0), fx=0.5, fy=0.5)
                            
                # --- NEW: VISUAL ODOMETRY ---
                small_mono = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                is_stuck = False
                if self.prev_frame is not None:
                    diff = cv2.absdiff(small_mono, self.prev_frame)
                    if np.mean(diff) < 1.0:
                        is_stuck = True
                self.prev_frame = small_mono
                self.stuck_pub.publish(is_stuck)

                h, w = small.shape[:2]
                fov = small[int(h*0.6):h, :]
                fov_h, fov_w = fov.shape[:2]
                
                gray = cv2.cvtColor(fov, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(fov, cv2.COLOR_BGR2HSV)
                
                blurred = cv2.medianBlur(gray, K_SIZE)
                _, road_mask = cv2.threshold(blurred, THRESH_VAL, 255, cv2.THRESH_BINARY_INV) 

                s_channel = hsv[:, :, 1]
                s_mask = cv2.inRange(s_channel, S_MIN, S_MAX)
                mask = cv2.bitwise_and(road_mask, s_mask)

                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                road_found = False
                target_x = fov_w / 2.0

                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_cnt) > 500:
                        M = cv2.moments(largest_cnt)
                        if M["m00"] > 0:
                            target_x = int(M["m10"] / M["m00"])
                            road_found = True
                            cv2.drawContours(fov, [largest_cnt], -1, (0, 255, 0), 2)

                # PUBLISH VISION DATA
                self.visible_pub.publish(road_found)
                if road_found:
                    error = (fov_w / 2.0) - target_x
                    normalized_error = error / (fov_w / 2.0)
                    self.error_pub.publish(normalized_error)
                else:
                    self.error_pub.publish(0.0)

                # DEBUG VISUALS
                cv2.circle(fov, (int(target_x), int(fov_h/2)), 5, (0, 0, 255), -1) 
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                cv2.imshow("Dirt Vision", np.hstack((fov, mask_bgr)))
            
            cv2.waitKey(1)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = DirtVisionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass