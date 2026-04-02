#!/usr/bin/env python3
import os
import sys
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
import tensorflow as tf

SHOW_DEBUG_VIEW = False

# Import local utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from clue_detection.model_utils import BoardProcessor, int_to_char

MODEL_PATH      = '/home/fizzer/ENPH-353-Competition/src/clue_detection/models/clue_reader_local.h5'
TEAM_ID         = 'TeamName'
PASSWORD        = 'password'
CAMERA_TOPIC    = 'B1/rrbot/camera1/image_raw'
PROCESS_EVERY_N = 3
CONFIRM_COUNT   = 3 

class ClueReaderNode:
    def __init__(self):
        rospy.init_node('clue_reader_node', anonymous=True)
        self.team_id         = TEAM_ID
        self.password        = PASSWORD
        self.process_every_n = PROCESS_EVERY_N
        self.frame_count     = 0

        self.last_raw_result    = None
        self.consecutive_count  = 0
        self.published_clue_ids = set()

        self.model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        if not self.model: rospy.logerr("Model not found!")

        self.bridge    = CvBridge()
        self.processor = BoardProcessor()
        self.score_pub = rospy.Publisher('/score_tracker', String, queue_size=10)

        rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo("[ClueReader] Debugging enabled. Look for 'Debug View' window.")

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0: return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError: return

        result = self.process_frame(cv_img)

        if not result:
            self.consecutive_count = 0
            self.last_raw_result = None
            return

        clue_type, clue_val = result
        current_id = self.clue_num(clue_type)
        combined = f"{clue_type}:{clue_val}"

        if combined == self.last_raw_result:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 1
            self.last_raw_result = combined

        if self.consecutive_count >= CONFIRM_COUNT:
            if current_id and current_id not in self.published_clue_ids:
                self.publish_clue(clue_type, clue_val)
                self.published_clue_ids.add(current_id)
                self.consecutive_count = 0 
                self.last_raw_result = None

    def process_frame(self, bgr_image):
        board_crop = self.detect_billboard(bgr_image)
        
        #debug
        if board_crop is None:
            if SHOW_DEBUG_VIEW:
                display_img = cv2.resize(bgr_image, (500, 300))
                cv2.putText(display_img, "NO BOARD DETECTED", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Debug View", display_img)
                cv2.waitKey(1)
            return None

        # Standardize for OCR
        board_crop = cv2.resize(board_crop, (500, 300), interpolation=cv2.INTER_CUBIC)

        if SHOW_DEBUG_VIEW:
            debug_img = board_crop.copy()
            cv2.rectangle(debug_img, (self.processor.top_x1, self.processor.top_y1), 
                          (self.processor.top_x2, self.processor.top_y2), (0, 255, 0), 2)
            cv2.rectangle(debug_img, (self.processor.bot_x1, self.processor.bot_y1), 
                          (self.processor.bot_x2, self.processor.bot_y2), (255, 0, 0), 2)
            cv2.imshow("Debug View", debug_img)
            cv2.waitKey(1)

        gray = cv2.cvtColor(board_crop, cv2.COLOR_BGR2GRAY)

        try:
            type_list, val_list = self.processor.segment_both(gray)
            c_type = self.classify_sequence(type_list)
            c_val  = self.classify_sequence(val_list)
            if c_type:
                rospy.loginfo(f"[Live] Type: {c_type} | Val: {c_val}")
        except: return None

        if c_type and c_val:
            return c_type, c_val.strip()
        return None

    def classify_sequence(self, char_list):
        if not char_list: return None
        res = ""
        for item in char_list:
            if isinstance(item, str) and item == "SPACE":
                res += " "
            else:
                inp = item.reshape(1, 32, 32, 1) / 255.0
                pred = self.model.predict(inp, verbose=0)
                res += int_to_char[np.argmax(pred)]
        return res

    def detect_billboard(self, bgr_image):
        img_h, img_w = bgr_image.shape[:2]
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([100, 100, 20]), np.array([140, 255, 255]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        valid_boards = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            #ignore bottom of screen
            if y < (img_h * 0.35):
                continue
                
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.5 or aspect_ratio > 3.5:
                continue
                
            valid_boards.append(cnt)

        if not valid_boards: return None

        #pick board in bottom of screen
        best = max(valid_boards, key=lambda c: cv2.boundingRect(c)[1])

        x, y, w, h = cv2.boundingRect(best)
        v_pad, h_pad = int(h * 0.02), int(w * 0.02)
        
        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, 0.02 * peri, True)
        
        if len(approx) == 4:
            src = self._order_points(np.float32([pt[0] for pt in approx]))
            dst = np.float32([[0,0], [w,0], [w,h], [0,h]])
            H, _ = cv2.findHomography(src, dst)
            if H is None: return None
            warp = cv2.warpPerspective(bgr_image, H, (w, h))
            return warp[v_pad:h-v_pad, h_pad:w-h_pad]
        
        return bgr_image[y:y+h, x:x+w]

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s, d = pts.sum(axis=1), np.diff(pts, axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
        return rect

    def clue_num(self, clue_type):
        clues = ['SIZE', 'VICTIM', 'CRIME', 'TIME',
        'PLACE', 'MOTIVE', 'WEAPON', 'BANDIT']
        normalised = clue_type.upper().strip()
        for i, c in enumerate(clues):
            if c in normalised:
                return i + 1
        rospy.logwarn(f"[ClueReader] clue_num: no match for '{clue_type}' (normalised: '{normalised}')")
        return None

    def publish_clue(self, clue_type, prediction):
        loc = self.clue_num(clue_type)
        if loc:
            msg = f"{self.team_id},{self.password},{loc},{prediction}"
            self.score_pub.publish(String(data=msg))
            rospy.loginfo(f"PUBLISHED: {msg}")

    def run(self): rospy.spin()

if __name__ == '__main__':
    ClueReaderNode().run()