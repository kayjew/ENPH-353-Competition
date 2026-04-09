#!/usr/bin/env python3
import rospy
import math
from std_msgs.msg import Bool, String, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

#microstate
class ActionState:
    IDLE = 0
    FOLLOWING_LINE = 1
    STOPPED = 2
    RECOVERY = 3
    KILLED = 4
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

        #Peek values
        self.clue_active = False
        self.clue_offset = 0.0
        self.nudge_factor = 0.6
        self.was_peeking = False
        self.clue_reader_ready = False
        self.debug = False

        # Peek unwind state
        self.peek_unwind_direction = 0.0
        self.peek_unwind_strength = 0.0    
        self.stable_frame_count = 0        
        self.STABLE_FRAMES_NEEDED = 8      
        
        # lost frame tracking
        self.lost_frame_counter = 0
        self.MAX_LOST_FRAMES = 150  # 30Hz, 150 frames, 5 sec
        
        #start course by pid
        self.pid_twist = Twist()
        self.dirt_visible = False
                
        # lost frame tracking
        self.lost_frame_counter = 0
        self.MAX_LOST_FRAMES = 150  # 30Hz,150 frames, 5 sec
        self.paved_stuck = False
        self.dirt_stuck = False
        self.stuck_timer_start = 0.0
        
        #start course by pid
        self.pid_twist = Twist()
        self.dirt_pid_twist = Twist()
        self.active_mode = "PID" 
        
        # subscribe/publish
        self.cmd_pub = rospy.Publisher("/B1/cmd_vel", Twist, queue_size=1)
        self.pub_score = rospy.Publisher("/score_tracker", String, queue_size=1)
        
        rospy.Subscriber("/vision/road_visible", Bool, self.visible_callback)
        rospy.Subscriber("/vision/ped_red", Bool, self.ped_red_callback)
        rospy.Subscriber("/control/pid_vel", Twist, self.pid_callback)
        rospy.Subscriber("/vision/dirt_visible", Bool, self.dirt_visible_callback)
        rospy.Subscriber("/control/dirt_pid_vel", Twist, self.dirt_pid_callback)
        rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        rospy.Subscriber("/vision/wide_road", Bool, self.wide_callback)
        rospy.Subscriber("/vision/pink_line", Bool, self.teleport_callback)
        # rospy.Subscriber("/control/il_vel", Twist, self.il_callback) 
        rospy.Subscriber("/clue/horizontal_offset", Float32, self.offset_callback)
        rospy.Subscriber("/clue/active_processing", Bool, self.clue_callback)
        rospy.Subscriber("/clue/status", String, self.clue_status_callback)
        rospy.Subscriber("/vision/paved_stuck", Bool, self.paved_stuck_callback)
        rospy.Subscriber("/vision/dirt_stuck", Bool, self.dirt_stuck_callback)
        
        rospy.loginfo(f"Brain Node Initialized. Starting control loop in Mode: {self.active_mode}")

    def visible_callback(self, msg):
        self.road_visible = msg.data

    def pid_callback(self, msg):
        self.pid_twist = msg

    def dirt_visible_callback(self, msg):
        self.dirt_visible = msg.data

    def dirt_pid_callback(self, msg):
        self.dirt_pid_twist = msg

    def ped_red_callback(self, msg):
        self.ped_red_detected = msg.data

    def clue_status_callback(self, msg):
        if msg.data == "READY":
            self.clue_reader_ready = True

    def clue_callback(self, msg):
        prev = self.clue_active
        self.clue_active = msg.data

        #Storing info about peeking to return
        if prev and not self.clue_active and not self.was_peeking:
            self.peek_unwind_direction = 1.0 if self.clue_offset > 0 else -1.0
            self.peek_unwind_strength  = min(abs(self.clue_offset), 1.0)
            self.stable_frame_count    = 0
            if self.debug:    
                rospy.loginfo(f"Store unpeek dir={self.peek_unwind_direction:.1f} strength={self.peek_unwind_strength:.2f}")

    def offset_callback(self, msg):
        self.clue_offset = msg.data
    
    def lidar_callback(self, msg):
        valid_ranges = [r for r in msg.ranges if 0.035 < r < .5]
        self.lidar_obstructed = len(valid_ranges) > 0

    def wide_callback(self, msg):
        self.road_wide = msg.data
    
    def teleport_callback(self, msg):
        self.teleport_line_detected = msg.data

    def paved_stuck_callback(self, msg):
        self.paved_stuck = msg.data

    def dirt_stuck_callback(self, msg):
        self.dirt_stuck = msg.data

        
    def teleport_to_dirt(self):
        rospy.loginfo("Requesting Gazebo Teleport...")
        rospy.wait_for_service('/gazebo/set_model_state', timeout=2.0)
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = 'B1' 
            # These are the exact coordinates from your tester script
            state_msg.pose.position.x = -4.5 
            state_msg.pose.position.y = -2.3
            state_msg.pose.position.z = 0.1 
            yaw = 0
            state_msg.pose.orientation.z = math.sin(yaw / 2.0)
            state_msg.pose.orientation.w = math.cos(yaw / 2.0)
            
            set_state(state_msg)
            rospy.loginfo("Gazebo Teleport successful!")
        except rospy.ServiceException as e:
            rospy.logerr(f"Teleport failed: {e}")
        except rospy.ROSException:
            rospy.logerr("Gazebo teleport service timed out.")

    def run(self):
        rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            out_twist = Twist()
            
            if self.action_state in [ActionState.FOLLOWING_LINE, ActionState.RECOVERY]:
                
                # --- 1. The 10-Second Visual Jam Failsafe ---
                currently_stuck = self.paved_stuck if self.active_mode == "PID" else self.dirt_stuck
                if currently_stuck:
                    if self.stuck_timer_start == 0.0:
                        self.stuck_timer_start = rospy.get_time() 
                    elif rospy.get_time() - self.stuck_timer_start > 10.0:
                        rospy.logerr("CRITICAL: Visually stuck for 10s. Run Terminated.")
                        self.action_state = ActionState.KILLED # <-- Locks the state
                else:
                    self.stuck_timer_start = 0.0 
                
                # --- 2. The 5-Second Lost Track Failsafe ---
                is_currently_visible = self.road_visible if self.active_mode == "PID" else self.dirt_visible
                if not is_currently_visible:
                    self.lost_frame_counter += 1
                else:
                    self.lost_frame_counter = 0
                    
                if self.lost_frame_counter >= self.MAX_LOST_FRAMES:
                    rospy.logerr("CRITICAL: Road lost for 5s. Run Terminated.")
                    self.action_state = ActionState.KILLED # <-- Locks the state
            
            elif self.action_state == ActionState.KILLED:
                pass # Do nothing. We are dead.
            else:
                # If we are IDLE or safely STOPPED, reset the counters
                self.stuck_timer_start = 0.0
                self.lost_frame_counter = 0
            
            
            # MACRO — Where are we on the track?
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
                            self.active_mode = "DIRT"
                            rospy.loginfo("Switched Brain routing to DIRT mode!")
                            self.course_section = CourseSection.TELEPORT_2
            elif self.course_section == CourseSection.TELEPORT_2:
                if self.action_state == ActionState.STOPPED:
                    self.action_state = ActionState.FOLLOWING_LINE
                    self.dash_start_time = rospy.get_time()
                if self.action_state == ActionState.FOLLOWING_LINE:
                    if rospy.get_time() - self.dash_start_time > 70:
                        if self.lidar_obstructed:
                           rospy.loginfo("Wall reached! Stopping for Teleport 3.")
                           self.action_state = ActionState.STOPPED
                           self.teleport_to_dirt()
                           self.course_section = CourseSection.FINISH_ZONE
                           self.dash_start_time = rospy.get_time()
                           
                           """ self.dash_start_time = rospy.get_time()
                            self.action_state = ActionState.IDLE"""
            elif self.course_section == CourseSection.FINISH_ZONE:
                if self.action_state == ActionState.STOPPED:
                    if rospy.get_time()-self.dash_start_time < 8:
                        self.action_state = ActionState.IDLE
                elif not self.lidar_obstructed:
                    self.action_state = ActionState.FOLLOWING_LINE
                    self.dash_start_time = rospy.get_time()
                else:
                    if rospy.get_time()-self.dash_start_time < .1:
                        self.action_state = ActionState.IDLE
                    
            # MICRO,What are the wheels doing?
            if self.action_state == ActionState.IDLE:
                if self.road_visible and self.clue_reader_ready and self.course_section == CourseSection.START_ZONE:
                    self.action_state = ActionState.FOLLOWING_LINE
                    self.course_section = CourseSection.PEDESTRIAN_CROSSING
                    self.pub_score.publish("rover,123,0,aaaaaa")
                    rospy.loginfo("Road seen! State: FOLLOWING_LINE")
                elif self.course_section == CourseSection.TELEPORT_1:
                    out_twist.linear.x = 0.3
                    out_twist.angular.z = 1.0
                elif self.course_section == CourseSection.FINISH_ZONE:
                    out_twist.linear.x = 1.0
                """elif self.course_section == CourseSection.TELEPORT_2:
                    if rospy.get_time() - self.dash_start_time < 2.5:
                        out_twist.angular.z = 1.0
                    if not self.teleport_line_detected:
                        out_twist.linear.x = 0.3
                    else:
                        self.course_section = CourseSection.TELEPORT_3
                        self.action_state = ActionState.STOPPED"""

            elif self.action_state == ActionState.FOLLOWING_LINE:
                if not (self.road_visible or self.dirt_visible):
                    self.action_state = ActionState.RECOVERY
                    rospy.logwarn("Road lost! State: RECOVERY")
                else:
                    if self.active_mode == "PID":

                        #peeking
                        if self.clue_active:
                            out_twist.linear.x = 0.02 if self.clue_offset > 0 else 0.03
                            
                            self.was_peeking = True
                            
                            #Store peeking info
                            self.peek_unwind_direction = -1.0 if self.clue_offset > 0 else 1.0
                            self.peek_unwind_strength = abs(self.clue_offset)

                            #Steering toward board
                            if abs(self.clue_offset) > 0.4:
                                raw = self.clue_offset * -1.2
                                out_twist.angular.z = max(-0.5, min(0.5, raw))
                            else:
                                out_twist.angular.z = (self.pid_twist.angular.z * 0.2) + (self.clue_offset * -1)
                            if self.debug:    
                                rospy.loginfo_throttle(0.2, f"Peek: {self.clue_offset:.2f}")
                        
                        # Undo peek
                        elif self.was_peeking:
                            unwind_angular = self.peek_unwind_direction * self.peek_unwind_strength * 1.2
                            out_twist.linear.x  = 0.15  
                            out_twist.angular.z = unwind_angular
                            self.peek_unwind_strength *= 0.92

                            #Prevent PID from going crazy
                            if self.peek_unwind_strength < 0.05:
                                if self.debug:
                                    rospy.loginfo("Undo peek")
                                self.was_peeking = False
                                self.peek_unwind_strength = 0.0
                            if self.debug:
                                rospy.loginfo_throttle(0.1, f"[Unwind] dir={self.peek_unwind_direction:.1f} str={self.peek_unwind_strength:.3f}")
                        else:
                            out_twist.linear.x = self.pid_twist.linear.x
                            out_twist.angular.z = self.pid_twist.angular.z
                        out_twist.linear.x = self.pid_twist.linear.x
                        out_twist.angular.z = self.pid_twist.angular.z
                    elif self.active_mode == "DIRT":
                        out_twist.linear.x = self.dirt_pid_twist.linear.x
                        out_twist.angular.z = self.dirt_pid_twist.angular.z

            elif self.action_state == ActionState.RECOVERY:
                out_twist.angular.z = 0.5  
                out_twist.linear.x = 0.0
                
                if self.active_mode == "PID" and self.road_visible:
                    self.action_state = ActionState.FOLLOWING_LINE
                    rospy.loginfo("Paved Road found! State: FOLLOWING_LINE")
                elif self.active_mode == "DIRT" and self.dirt_visible:
                    self.action_state = ActionState.FOLLOWING_LINE
                    rospy.loginfo("Dirt Road found! State: FOLLOWING_LINE")
            
            elif self.action_state == ActionState.STOPPED:
                out_twist.linear.x = 0.0
                out_twist.angular.z = 0.0

            elif self.action_state in [ActionState.STOPPED, ActionState.KILLED]:
                # Fully cut the motors
                self.pub_score.publish("rover,123,-1,aaaaaa")
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