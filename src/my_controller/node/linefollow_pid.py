#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
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
        if dt <= 0.0:
            dt = 0.001 
            
        self.integral += error * dt
        
        # Anti-windup safety clamp
        self.integral = max(-5.0, min(5.0, self.integral))
        
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        self.prev_error = error
        self.last_time = current_time
        return output

class LineFollowPIDNode:
    def __init__(self):
        rospy.init_node('linefollow_pid')
        
        self.pid = PIDController(kp=40, ki=0.0, kd=0.5)
        self.current_error = 0.0
        self.target_speed = 1
        
        #publisher/subscriber
        rospy.Subscriber("/vision/lane_error", Float32, self.error_callback)
        self.cmd_pub = rospy.Publisher("/control/pid_vel", Twist, queue_size=1)
        
        rospy.loginfo("Line Follow PID Node Initialized.")

    def error_callback(self, msg):
        self.current_error = msg.data

    def run(self):
        # Run at 30 Hz to match the camera frame rate
        rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            twist = Twist()
            twist.linear.x = self.target_speed
            
            # Compute steering based on the latest error
            raw_steering = self.pid.compute(self.current_error)
            
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