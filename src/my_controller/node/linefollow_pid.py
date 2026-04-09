#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Twist

class PIDController:
    """A simple PID controller for smooth steering."""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.last_time = 0.0

    def compute(self, error):
        current_time = rospy.get_time() 
        if self.last_time == 0.0:
            self.last_time = current_time
            return 0.0
        dt = current_time - self.last_time
        if dt <= 0.0: dt = 0.001 
            
        self.integral += error * dt
        self.integral = max(-5.0, min(5.0, self.integral))
        
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        self.last_time = current_time
        return output

class LineFollowPIDNode:
    def __init__(self):
        rospy.init_node('linefollow_pid')
        
        self.pid = PIDController(kp=65, ki=0.4, kd=0.15)
        self.current_error = 0.0
        self.target_speed = 1
        
        #Introduce bias in pid
        self.CLUE_BIASES = [0.1, 0.1, -0.05, -0.15, 0.1, 0.1, -0.1, 0.1]
        self.current_bias = self.CLUE_BIASES[0] 
        
        # subscribers
        rospy.Subscriber("/score_tracker", String, self.clue_submitted_callback)
        rospy.Subscriber("/vision/lane_error", Float32, self.error_callback)
        
        self.cmd_pub = rospy.Publisher("/control/pid_vel", Twist, queue_size=1)
        
        rospy.loginfo("Line Follow PID Node Initialized with ID-Sync Biasing.")

    def error_callback(self, msg):
        self.current_error = msg.data

    def clue_submitted_callback(self, msg):
        """Parses the board ID from the submission string and updates bias."""
        try:
            # submission format: "team_id,password,board_num,value"
            parts = msg.data.split(',')
            if len(parts) >= 3:
                board_id = int(parts[2]) # Get the 3rd element
                next_idx = board_id 
                
                if next_idx < len(self.CLUE_BIASES):
                    self.current_bias = self.CLUE_BIASES[next_idx]
                    rospy.loginfo(f"Board {board_id} cleared. Biasing for Board {board_id + 1}: {self.current_bias}")
                else:
                    self.current_bias = 0.0 # Center if all 8 are done
                    rospy.loginfo("All boards cleared")
        except Exception as e:
            rospy.logwarn(f"Could not parse {e}")

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            twist = Twist()
            twist.linear.x = self.target_speed
            
            # Apply the bias to the error
            biased_error = self.current_error + self.current_bias
            
            raw_steering = self.pid.compute(biased_error)
            
            # Clamp the steering to prevent sudden violent jerks
            twist.angular.z = max(-2, min(2, raw_steering))
            
            self.cmd_pub.publish(twist)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = LineFollowPIDNode()
        node.run()
    except rospy.ROSInterruptException:
        pass