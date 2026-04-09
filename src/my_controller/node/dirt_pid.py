#!/usr/bin/env python3
import rospy
import time
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error, self.integral = 0, 0
        self.last_time = 0.0

    def compute(self, error):
        current_time = rospy.get_time() 
        if self.last_time == 0.0:
            self.last_time = current_time
            return 0.0
        dt = current_time - self.last_time
        if dt <= 0.0: dt = 0.001 
        self.integral = max(-5.0, min(5.0, self.integral + error * dt))
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error, self.last_time = error, current_time
        return output

class DirtPIDNode:
    def __init__(self):
        rospy.init_node('dirt_pid')
        # YOUR DIRT TUNINGS
        self.pid = PIDController(kp=40.0, ki=0.0, kd=0.1)
        self.current_error = 0.0
        self.target_speed = 0.4
        
        rospy.Subscriber("/vision/dirt_error", Float32, self.error_callback)
        self.cmd_pub = rospy.Publisher("/control/dirt_pid_vel", Twist, queue_size=1)
        rospy.loginfo("Dirt PID Node Initialized.")

    def error_callback(self, msg):
        self.current_error = msg.data

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            twist = Twist()
            twist.linear.x = self.target_speed
            raw_steering = self.pid.compute(self.current_error)
            twist.angular.z = max(-2, min(2, raw_steering))
            self.cmd_pub.publish(twist)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = DirtPIDNode()
        node.run()
    except rospy.ROSInterruptException:
        pass