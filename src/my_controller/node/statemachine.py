#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


#microstate
class ActionState:
    IDLE = 0
    FOLLOWING_LINE = 1
    STOPPED = 2
    RECOVERY = 3
    DASHING = 4
#macrostate
class CourseSection:
    START_ZONE = 0
    PEDESTRIAN_CROSSING = 1
    ROUND_ABOUT = 2
    TELEPORT_1 = 3
    TELEPORT_2 = 4
    TELEPORT_3 = 5
    FINISH_ZONE = 6

class BrainNode:
    def __init__(self):
        rospy.init_node('brain_node')
        
        #set initial micro/macro state
        self.action_state = ActionState.IDLE
        self.course_section = CourseSection.START_ZONE
        self.road_visible = False
        self.ped_red_detected = False
        self.prev_lidar_reading = False
        self.lidar_obstructed = False
        self.road_wide = False
        self.lidar_error = 0.0
        self.wide_frame_count = 0
        self.dash_start_time = 0.0
        self.exited_roundabout = False
        self.teleport_line_detected = False
        
        # lost frame tracking
        self.lost_frame_counter = 0
        self.MAX_LOST_FRAMES = 150  # 30Hz,150 frames, 5 sec
        
        #start course by pid
        self.pid_twist = Twist()
        #self.il_twist = Twist()
        self.active_mode = "PID" 
        
        # subscribe/publish
        self.cmd_pub = rospy.Publisher("/B1/cmd_vel", Twist, queue_size=1)
        self.pub_score = rospy.Publisher("/score_tracker", String, queue_size=1)
        
        rospy.Subscriber("/vision/road_visible", Bool, self.visible_callback)
        rospy.Subscriber("/vision/ped_red", Bool, self.ped_red_callback)
        rospy.Subscriber("/control/pid_vel", Twist, self.pid_callback)
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        rospy.Subscriber("/vision/wide_road", Bool, self.wide_callback)
        rospy.Subscriber("/vision/pink_line", Bool, self.teleport_callback)
        # rospy.Subscriber("/control/il_vel", Twist, self.il_callback) 
        
        rospy.loginfo(f"Brain Node Initialized. Starting control loop in Mode: {self.active_mode}")

    def visible_callback(self, msg):
        self.road_visible = msg.data

    def pid_callback(self, msg):
        self.pid_twist = msg

    def ped_red_callback(self, msg):
        self.ped_red_detected = msg.data
    
    def lidar_callback(self, msg):
        valid_ranges = [r for r in msg.ranges if 0.035 < r < .5]
        self.lidar_obstructed = len(valid_ranges) > 0

    def wide_callback(self, msg):
        self.road_wide = msg.data
    
    def teleport_callback(self, msg):
        self.teleport_line_detected = msg.data

        
    #def il_callback(self, msg):
    #    self.il_twist = msg

    def run(self):
        rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            out_twist = Twist()
            
            # lost frame tracking
            if not self.road_visible:
                self.lost_frame_counter += 1
            else:
                self.lost_frame_counter = 0
                
            if self.lost_frame_counter >= self.MAX_LOST_FRAMES and self.action_state != ActionState.STOPPED:
                rospy.logerr(f"Road lost for {self.MAX_LOST_FRAMES} frames! Triggering FULL STOP.")
                self.action_state = ActionState.STOPPED
            
            
            # MACRO,Where are we on the track?
            if self.course_section == CourseSection.START_ZONE:
                pass
                
            elif self.course_section == CourseSection.PEDESTRIAN_CROSSING:
                if self.action_state == ActionState.FOLLOWING_LINE and self.ped_red_detected:
                    rospy.loginfo("Red line detected. Stopping.")
                    self.action_state = ActionState.STOPPED
                    self.prev_lidar_reading = False
                elif self.action_state == ActionState.STOPPED:
                    if self.prev_lidar_reading == True and self.lidar_obstructed == False:
                        rospy.loginfo("FALLING EDGE DETECTED: Pedestrian has passed.")
                        self.course_section = CourseSection.ROUND_ABOUT
                        self.action_state = ActionState.FOLLOWING_LINE
                    self.prev_lidar_reading = self.lidar_obstructed
                pass

            elif self.course_section == CourseSection.ROUND_ABOUT:
                if self.road_wide:
                    self.wide_frame_count += 1
                    rospy.loginfo(f"Wide road detected! Count is: {self.wide_frame_count}")
                    if self.wide_frame_count > 15:
                        rospy.loginfo("Road is wide. Entering Roundabout.")
                        self.action_state = ActionState.STOPPED
                        self.course_section = CourseSection.TELEPORT_1
                        self.prev_lidar_reading = False
                        self.wide_frame_count =0
                else:
                    self.wide_frame_count =0

            elif self.course_section == CourseSection.TELEPORT_1:
                if self.action_state == ActionState.STOPPED:
                    if self.prev_lidar_reading == True and self.lidar_obstructed == False:
                        rospy.loginfo("Gap in traffic! Starting left turn.")
                        self.action_state=ActionState.IDLE
                        self.dash_start_time = rospy.get_time()
                    self.prev_lidar_reading = self.lidar_obstructed
                elif self.action_state == ActionState.IDLE:
                    if rospy.get_time() - self.dash_start_time > 1.5:
                        rospy.loginfo("Turn complete. Resuming Line Follow.") 
                        self.action_state = ActionState.FOLLOWING_LINE 
                elif self.action_state == ActionState.FOLLOWING_LINE:
                     if not self.exited_roundabout:
                        if self.road_wide:
                            self.wide_frame_count += 1
                            rospy.loginfo(f"Wide road detected! Count is: {self.wide_frame_count}")
                            if self.wide_frame_count > 10:
                                rospy.loginfo("Road is wide. Exiting Roundabout.")
                                self.exited_roundabout = True
                                self.action_state = ActionState.IDLE
                                self.dash_start_time = rospy.get_time()
                                self.prev_lidar_reading = False
                                self.wide_frame_count =0
                        else:
                            self.wide_frame_count =0
                     else:
                         # We are still line following perfectly, just checking a different sensor
                        if self.teleport_line_detected:
                            rospy.loginfo("Pink Line detected! Stopping and moving to TELEPORT_2.")
                            self.action_state = ActionState.STOPPED
                            self.course_section = CourseSection.TELEPORT_2

            
            # MICRO,What are the wheels doing?
            if self.action_state == ActionState.IDLE:
                if self.road_visible and self.course_section == CourseSection.START_ZONE:
                    self.action_state = ActionState.FOLLOWING_LINE
                    self.course_section = CourseSection.PEDESTRIAN_CROSSING
                    #start timer
                    self.pub_score.publish("rover,123,0,aaaaaa")
                    rospy.loginfo("Road seen! State: FOLLOWING_LINE")
                elif self.course_section == CourseSection.TELEPORT_1:
                    out_twist.linear.x = 0.3
                    out_twist.angular.z = 1.0

            elif self.action_state == ActionState.FOLLOWING_LINE:
                if not self.road_visible:
                    self.action_state = ActionState.RECOVERY
                    rospy.logwarn("Road lost! State: RECOVERY")
                else:
                    if self.active_mode == "PID":
                        out_twist.linear.x = self.pid_twist.linear.x
                        out_twist.angular.z = self.pid_twist.angular.z
                    #elif self.active_mode == "IL":
                    #    out_twist = self.il_twist

            elif self.action_state == ActionState.RECOVERY:
                out_twist.angular.z = 0.5  
                out_twist.linear.x = 0.0
                
                if self.road_visible:
                    self.action_state = ActionState.FOLLOWING_LINE
                    rospy.loginfo("Road found! State: FOLLOWING_LINE")
            
            elif self.action_state == ActionState.STOPPED:
                # Fully cut the motors
                out_twist.linear.x = 0.0
                out_twist.angular.z = 0.0

            # Publish the final decision to the actual robot wheels
            self.cmd_pub.publish(out_twist)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = BrainNode()
        node.run()
    except rospy.ROSInterruptException:
        pass