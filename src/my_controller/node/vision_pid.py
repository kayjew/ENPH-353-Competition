#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

class PIDVisionNode:
    def __init__(self):
        rospy.init_node('vision_pid')
        self.bridge = CvBridge()
        
        # Publishers
        self.error_pub = rospy.Publisher("/vision/lane_error", Float32, queue_size=1)
        self.visible_pub = rospy.Publisher("/vision/road_visible", Bool, queue_size=1)
        self.wide_road_pub = rospy.Publisher("/vision/wide_road", Bool, queue_size=1)
        self.ped_pub = rospy.Publisher("/vision/ped_red", Bool, queue_size=1)
        self.tel_pub = rospy.Publisher("/vision/pink_line", Bool, queue_size=1)
        
        # Subscriber
        self.image_sub = rospy.Subscriber("/B1/rrbot/camera1/image_raw", Image, self.image_callback, queue_size=1)
        
        rospy.loginfo("PID Vision Node Initialized")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            height, width, _ = cv_image.shape
            
            #choose FOV to save time processing
            fov = cv_image[int(height*0.6) : height, :] 
            fov_height = fov.shape[0]
            
            # Convert to HSV and threshold for the road color
            hsv = cv2.cvtColor(fov, cv2.COLOR_BGR2HSV)
            lower_gray = np.array([0, 0, 50])
            upper_gray = np.array([180, 50, 200])
            mask = cv2.inRange(hsv, lower_gray, upper_gray)

            #threshold for pedestrian crossing
            lower_ped = np.array([0, 50, 50]) 
            upper_ped = np.array([10, 200, 200])
            ped_mask = cv2.inRange(hsv, lower_ped, upper_ped)
            
            ped_contours, _ = cv2.findContours(ped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            is_ped = False

            if ped_contours:
                largest_ped = max(ped_contours, key=cv2.contourArea)
                if cv2.contourArea(largest_ped) > 500: # Ensure it's not just a noisy pixel
                    is_ped = True
            
            self.ped_pub.publish(is_ped)

            #threshold for teleport
            lower_tel = np.array([150,80, 50]) 
            upper_tel = np.array([180, 150, 150])
            tel_mask = cv2.inRange(hsv, lower_tel, upper_tel)
            
            tel_contours, _ = cv2.findContours(tel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            is_tel = False

            if tel_contours:
                largest_tel = max(tel_contours, key=cv2.contourArea)
                if cv2.contourArea(largest_tel) > 500: # Ensure it's not just a noisy pixel
                    is_tel = True
            
            self.tel_pub.publish(is_tel)

            #find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.visible_pub.publish(False)
                self.error_pub.publish(0.0) 
                return
                
            # Find the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            #create the final road mask for all future analysis
            final_road = np.zeros_like(mask)
            cv2.drawContours(final_road, [largest_contour], -1, 255, thickness=cv2.FILLED)

            #find scanline position
            x, y, w, h = cv2.boundingRect(largest_contour)
            scanline_row = int(min(y + h*0.10, fov_height - 1))

            # filter scanline for road indices
            row_data = final_road[scanline_row, :]
            
            road_indices = np.where(row_data == 255)[0]

            if len(road_indices) == 0:
                self.visible_pub.publish(False)
                self.error_pub.publish(0.0) 
                return
            
            self.visible_pub.publish(True)

            left_edge = road_indices[0]
            right_edge = road_indices[-1]

            midpoint = (left_edge + right_edge) / 2.0

            is_wide = (left_edge <= 0) and (right_edge >= width-1) 
            self.wide_road_pub.publish(is_wide)
            
            # The error is the distance from the camera center to this midpoint.
            raw_error = (width / 2.0) - midpoint
            
            normalized_error = raw_error / (width / 2.0)
            self.error_pub.publish(normalized_error)
            
            # debug
            cv2.circle(fov, (int(midpoint), scanline_row), 5, (0, 0, 255), -1) 
            cv2.line(fov, (0, scanline_row), (width, scanline_row), (0, 255, 0), 1)
            cv2.imshow("Geometric Vision", fov)
            cv2.waitKey(1)
            
            
                
        except Exception as e:
            rospy.logerr(f"Vision error: {e}")

if __name__ == '__main__':
    try:
        node = PIDVisionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass