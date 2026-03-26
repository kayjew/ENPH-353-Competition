#!/usr/bin/env python3
"""

Message format:
    "TEAM_ID,PASSWORD,CLUE_LOCATION,CLUE_PREDICTION"
    e.g. "TeamRed,multi21,2,JEDIS"

Usage:
    rosrun clue_detection clue_reader_node.py \
        _model_path:=/path/to/clue_reader_model.h5 \
        _team_id:=TeamName \
        _password:=mypass \
        _clue_location:=1
"""

import os
import sys
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from clue_detection.model_utils import BoardProcessor, int_to_char, CHARS
import tensorflow as tf


class ClueReaderNode:
    def __init__(self):
        rospy.init_node('clue_reader_node', anonymous=True)

        # ROS commands from command line
        model_path     = rospy.get_param('~model_path',    '')
        self.team_id   = rospy.get_param('~team_id',       'TeamName')
        self.password  = rospy.get_param('~password',      'password')
        self.clue_loc  = rospy.get_param('~clue_location', 1)
        camera_topic   = rospy.get_param('~camera_topic',  '/B1/pi_camera/image_raw')

    
        self.process_every_n = rospy.get_param('~process_every_n', 10)
        self.frame_count = 0

        # Track past
        self.last_published = None

        # Import model
        self.model = None
        if model_path and os.path.exists(model_path):
            rospy.loginfo(f"[ClueReader] Loading model from: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            rospy.loginfo("[ClueReader] Model loaded successfully.")
        else:
            rospy.logwarn("[ClueReader] No model found. Set _model_path:=... to enable predictions.")

        # Util
        self.bridge    = CvBridge()
        self.processor = BoardProcessor()

        # Pub
        self.score_pub = rospy.Publisher('/score_tracker', String, queue_size=10)

        # Sub
        rospy.Subscriber(camera_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo(f"[ClueReader] Subscribed to: {camera_topic}")
        rospy.loginfo("[ClueReader] Node ready.")

    #Camera callback
    def image_callback(self, msg):
        # Throttle processing to every N frames
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"[ClueReader] cv_bridge error: {e}")
            return

        result = self.process_frame(cv_image)
        if result is not None:
            clue_type, clue_value = result
            prediction = f"{clue_type}_{clue_value}"

            # Only publish if we have a new prediction
            if prediction != self.last_published:
                self.publish_clue(prediction)
                self.last_published = prediction

    #Billboard detection
    def detect_billboard(self, bgr_image):
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # Dark blue border
        lower = np.array([100, 180, 40])
        upper = np.array([130, 255, 160])
        mask = cv2.inRange(hsv, lower, upper)

        # Clean up mask
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best) < 5000:
            return None

        x, y, w, h = cv2.boundingRect(best)

        # Strip the blue border to get just the gray interior
        border = int(min(w, h) * 0.08)
        ix = x + border
        iy = y + border
        iw = w - 2 * border
        ih = h - 2 * border

        if iw < 50 or ih < 50:
            return None

        return bgr_image[iy:iy + ih, ix:ix + iw]

    # Frame processing
    def process_frame(self, bgr_image):

        if self.model is None:
            return None

        #Detect billboard
        board_crop = self.detect_billboard(bgr_image)
        if board_crop is None:
            return None

        gray = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY)

        #Process board
        type_imgs, val_imgs = self.processor.segment_both(gray)

        if not type_imgs or not val_imgs:
            return None

        #Identify chars
        clue_type  = self.classify_chars(type_imgs)
        clue_value = self.classify_chars(val_imgs)

        if clue_type and clue_value:
            rospy.loginfo(f"[ClueReader] Detected: type={clue_type}  value={clue_value}")
            return clue_type, clue_value

        return None
    
    #Runs NN on image and returns string if valid or none
    def classify_chars(self, char_imgs):
        processed = []
        for img in char_imgs:
            if img is None:
                return None
            processed.append(img.reshape(32, 32, 1) / 255.0)

        batch = np.array(processed)
        preds = self.model.predict(batch, verbose=0)
        chars = [int_to_char[np.argmax(p)] for p in preds]
        return ''.join(chars)

    #Publisher
    def publish_clue(self, prediction):
        """
        Formats and publishes the score_tracker message.
        Format: TEAM_ID,PASSWORD,CLUE_LOCATION,CLUE_PREDICTION
        """
        msg_str = f"{self.team_id},{self.password},{self.clue_loc},{prediction}"
        self.score_pub.publish(String(data=msg_str))
        rospy.loginfo(f"[ClueReader] Published: {msg_str}")

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = ClueReaderNode()
        node.run()
    except rospy.ROSInterruptException:
        pass