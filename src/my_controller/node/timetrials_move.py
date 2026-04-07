#! /usr/bin/env python3

import rospy
import cv2
import numpy as np
import sys

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class linefollowing:
    def __init__(self):
       self.bridge = CvBridge()

       self.lost_line_counter = 0
       self.finish_triggered = False
       self.MAX_LOST_FRAMES = 10

       #Set-up publisher subscriber relationship
       self.pub_cmd = rospy.Publisher('/B1/cmd_vel', Twist, queue_size=1)
       self.pub_score = rospy.Publisher('/score_tracker',String,queue_size=1)
       self.image_sub = rospy.Subscriber('B1/rrbot/camera1/image_raw',Image,self.callback)

       #start timer
       rospy.sleep(1)
       self.pub_score.publish("rover,123,0,aaaaaa")

    def callback(self,data):
       if self.finish_triggered:
            return
       
       cmd = Twist()
       p=150

       #Attempts to convert image to something readable by cv2, if not possible throws 'e' for error
       try:
         cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
       except CvBridgeError as e:
         print(e)

       height, width, _ = cv_image.shape
   
       #Threshold image and define FOV
       blurred = cv2.medianBlur(cv_image, 5)
       hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
       lower_thresh = np.array([0, 0, 0])    # H, S, V
       upper_thresh= np.array([150, 50, 255]) # Low saturation limit is key here
       img_bin = cv2.inRange(hsv, lower_thresh, upper_thresh)
       
       FOV = img_bin[int(0.9*height):height, :]
       
       #debug
       cv2.imshow("Original Camera", cv_image)
       cv2.imshow("Thresholded Image", img_bin)
       cv2.waitKey(1)

       #Find max contours
       contours, hierarchy = cv2.findContours(FOV, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
       if len(contours) == 0:
            self.lost_line_counter += 1

            if self.lost_line_counter > self.MAX_LOST_FRAMES:
              if self.finish_triggered != True:
                cmd.linear.x = 0
                cmd.angular.z = 0
                self.pub_cmd.publish(cmd)
                self.finish_triggered = True
                self.pub_score.publish("rover,123,-1,aaaaaa")
                return
              return

            cmd.linear.x = 0
            cmd.angular.z = .5
            self.pub_cmd.publish(cmd)
            return
                
       else:
            self.lost_line_counter = 0
            largest = max(contours, key=cv2.contourArea)
            moment = cv2.moments(largest)
            
       if moment["m00"]==0:
            self.lost_line_counter += 1

            if self.lost_line_counter > self.MAX_LOST_FRAMES:
              if self.finish_triggered != True:
                cmd.linear.x = 0
                cmd.angular.z = 0
                self.pub_cmd.publish(cmd)
                self.finish_triggered = True
                self.pub_score.publish("rover,123,-1,aaaaaa")
                return
              return

            cmd.linear.x = 0
            cmd.angular.z = .5
            self.pub_cmd.publish(cmd)
            return
       else:
            centriod_x = int(moment["m10"] / moment["m00"])
            centriod_y = int(moment["m01"] / moment["m00"])
            centriod_y += int(0.9 * height)
       

       allign_to_centre = centriod_x-(width/2)

       cmd.linear.x = .2
       cmd.angular.z = -allign_to_centre/p

       try:
          self.pub_cmd.publish(cmd)
       except CvBridgeError as e:
          print(e)

def main(args):
   rospy.init_node('timetrials_move', anonymous=True)
   lf = linefollowing()
   rospy.spin()
   

if __name__ == '__main__':
    main(sys.argv)          
