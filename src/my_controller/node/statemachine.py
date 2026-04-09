#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool, String, Float32
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
        self.pedestrian_in_way = False
        self.lidar_error = 0.0
        self.clue_active = False
        self.clue_offset = 0.0
        self.nudge_factor = 0.6
        self.was_peeking = False
        
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
        rospy.Subscriber("/clue/horizontal_offset", Float32, self.offset_callback)
        rospy.Subscriber("/clue/active_processing", Bool, self.clue_callback)
        # rospy.Subscriber("/control/il_vel", Twist, self.il_callback) 
        
        rospy.loginfo(f"Brain Node Initialized. Starting control loop in Mode: {self.active_mode}")

    def visible_callback(self, msg):
        self.road_visible = msg.data

    def pid_callback(self, msg):
        self.pid_twist = msg

    def ped_red_callback(self, msg):
        self.ped_red_detected = msg.data

    def clue_callback(self, msg):
        self.clue_active = msg.data

    def offset_callback(self, msg):
        self.clue_offset = msg.data
    
    def lidar_callback(self, msg):
        front_cone = list(msg.ranges[-20:]) + list(msg.ranges[:20])
        valid_ranges = [r for r in front_cone if 0.1 < r < 10.0]
        
        # If any object is within 0.8m of the front, path is blocked
        if valid_ranges and min(valid_ranges) < 0.8:
            self.pedestrian_in_way = True
        else:
            self.pedestrian_in_way = False

        # 2. Widest Gap Strategy (For Roundabout navigation)
        # Look at 180 degree arc in front (indices 180 to 540)
        front_arc = msg.ranges[180:540]
        if front_arc:
            # Find the direction of the "deepest" opening
            max_val = max(front_arc)
            if max_val > 0:
                max_idx = front_arc.index(max_val)
                # Normalize to [-1.0 (left), 1.0 (right)]
                self.lidar_error = (180 - max_idx) / 180.0

        
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
                    rospy.loginfo("Red line detected! Stopping for crosswalk.")
                    self.action_state = ActionState.STOPPED
                    #self.dash_timer_start = 0.0 # Reset the dash timer
                # Waiting Phase
                elif self.action_state == ActionState.STOPPED:
                    if not self.pedestrian_in_way:
                        rospy.loginfo("Path clear! DASHING across crosswalk.")
                        self.action_state = ActionState.DASHING
                        self.dash_timer_start = rospy.get_time()
                
                # Dash Phase (Cross the street regardless of vision for 2 seconds)
                elif self.action_state == ActionState.DASHING:
                    if (rospy.get_time() - self.dash_timer_start) > 2.0:
                        rospy.loginfo("Crosswalk cleared. Entering ROUND_ABOUT.")
                        self.course_section = CourseSection.ROUND_ABOUT
                        self.action_state = ActionState.FOLLOWING_LINE
                pass

            
            # MICRO,What are the wheels doing?
            if self.action_state == ActionState.IDLE:
                if self.road_visible and self.course_section == CourseSection.START_ZONE:
                    self.action_state = ActionState.FOLLOWING_LINE
                    self.course_section = CourseSection.PEDESTRIAN_CROSSING
                    #start timer
                    self.pub_score.publish("rover,123,0,aaaaaa")
                    rospy.loginfo("Road seen! State: FOLLOWING_LINE")

            elif self.action_state == ActionState.FOLLOWING_LINE:
                if not self.road_visible:
                    self.action_state = ActionState.RECOVERY
                    rospy.logwarn("Road lost! State: RECOVERY")
                else:
                   if self.active_mode == "PID":
                        if self.clue_active:
                            if self.clue_offset > 0:
                                out_twist.linear.x = 0.02
                            else:
                                out_twist.linear.x = 0.03
                            self.was_peeking = True
                            
                            if abs(self.clue_offset) > 0.4:
                                out_twist.angular.z = self.clue_offset * -1.2 
                            else:
                                out_twist.angular.z = (self.pid_twist.angular.z * 0.2) + (self.clue_offset * -1)
                                
                            rospy.loginfo_throttle(0.2, f"HEAVY PEEK: {self.clue_offset:.2f}")
                        
                        elif getattr(self, 'was_peeking', False):
                            out_twist.linear.x = 0.12 
                            out_twist.angular.z = self.pid_twist.angular.z * 1.6
                            if abs(self.pid_twist.angular.z) < 0.15:
                                rospy.loginfo("Alignment clear. Resuming full speed.")
                                self.was_peeking = False
                        
                        else:
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
            
            elif self.action_state == ActionState.DASHING:
                out_twist.linear.x = 0.8 # Dash speed
                out_twist.angular.z = 0.0

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