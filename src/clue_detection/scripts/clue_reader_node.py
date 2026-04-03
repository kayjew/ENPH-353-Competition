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

SHOW_DEBUG_VIEW = True
MODEL_PATH      = '/home/fizzer/ENPH-353-Competition/src/clue_detection/models/clue_reader_local.h5'
TEAM_ID         = 'TeamName'
PASSWORD        = 'password'
CAMERA_TOPIC    = 'B1/rrbot/camera1/image_raw'
PROCESS_EVERY_N = 1
CONFIRM_COUNT   = 3 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from clue_detection.model_utils import BoardProcessor, int_to_char

def get_stable_string(raw_str):
    #Transforms problem strings
    if not raw_str:
        return ""

    mapping = {
        '1': 'I', '0': 'O', '5': 'S', '7': 'T', 
        '4': 'A', '3': 'E', '8': 'B', '9': 'M'
    }
    
    normalized = "".join([mapping.get(c, c) for c in raw_str.upper()])
    
    keywords = [
        "MOTIVE", "VICTIM", "SIZE", "PLACE", "CRIME", "TIME", "WEAPON", "BANDIT"]
    
    for word in keywords:
        if word in normalized:
            return word
            
    return "".join([c for c in normalized if c.isalnum()])

class ClueReaderNode:
    def __init__(self):
        rospy.init_node('clue_reader_node', anonymous=True)
        self.team_id         = TEAM_ID
        self.password        = PASSWORD
        self.process_every_n = PROCESS_EVERY_N
        self.frame_count     = 0

        self.last_val           = None
        self.last_type_id       = None
        self.consecutive_count  = 0
        self.published_clue_ids = set()

        self.model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        if not self.model: 
            rospy.logerr("Model not found!")

        self.bridge    = CvBridge()
        self.processor = BoardProcessor()
        self.score_pub = rospy.Publisher('/score_tracker', String, queue_size=10)

        rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo("[ClueReader] Clue finding active")

    def clue_num(self, clue_type_str):
        clues = ['SIZE', 'VICTIM', 'CRIME', 'TIME', 'PLACE', 'MOTIVE', 'WEAPON', 'BANDIT']
        for i, c in enumerate(clues):
            if c in clue_type_str.upper():
                return i + 1
        return None

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0: return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError: return

        result = self.process_frame(cv_img)

        if not result:
            self.consecutive_count = 0
            return

        raw_type, raw_val = result
        
        current_type_str = get_stable_string(raw_type)
        current_val      = get_stable_string(raw_val)
        current_id       = self.clue_num(current_type_str)

        if current_id and current_val and current_id == self.last_type_id and current_val == self.last_val:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 1
            self.last_type_id = current_id
            self.last_val = current_val

        if self.consecutive_count == CONFIRM_COUNT:
            if current_id not in self.published_clue_ids:
                self.publish_clue(current_id, current_val)
                self.published_clue_ids.add(current_id)
                rospy.loginfo(f"STABLE MATCH: ID {current_id} ({current_type_str}) -> {current_val}")
                
                # Reset after publishing
                self.consecutive_count = 0 
                self.last_val = None
                self.last_type_id = None

    def process_frame(self, bgr_image):
        board_crop = self.detect_billboard(bgr_image)
        
        if board_crop is None:
            if SHOW_DEBUG_VIEW:
                display_img = cv2.resize(bgr_image, (500, 300))
                cv2.putText(display_img, "NO BOARD", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Debug View", display_img)
                cv2.waitKey(1)
            return None

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
                rospy.loginfo(f"[Live OCR] Raw Type: {c_type} | Raw Val: {c_val}")
            return c_type, c_val
        except: 
            return None

    def classify_sequence(self, char_list):
        if not char_list: return ""
        res = ""
        for item in char_list:
            if isinstance(item, str) and item == "SPACE":
                res += " "
            else:
                inp = item.reshape(1, 32, 32, 1) / 255.0
                pred = self.model.predict(inp, verbose=0)
                res += int_to_char[np.argmax(pred)]
        
        words = res.split()
        filtered_words = [w for w in words if len(w) > 1 or w.upper() in ["A", "I"]]
        return " ".join(filtered_words) if filtered_words else ""

    def detect_billboard(self, bgr_image):
        img_h, img_w = bgr_image.shape[:2]
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([100, 100, 20]), np.array([140, 255, 255]))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        valid_boards = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 6000: continue
            x, y, w, h = cv2.boundingRect(cnt)
            if y < (img_h * 0.35): continue
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.5 or aspect_ratio > 3.5: continue
            valid_boards.append(cnt)

        if not valid_boards: return None
        valid_boards.sort(key=cv2.contourArea, reverse=True)
        
        if len(valid_boards) > 1:
            area1 = cv2.contourArea(valid_boards[0])
            area2 = cv2.contourArea(valid_boards[1])
            if abs(area1 - area2) / area1 < 0.15:
                best = max(valid_boards[:2], key=lambda c: cv2.boundingRect(c)[1])
            else:
                best = valid_boards[0]
        else:
            best = valid_boards[0]

        x, y, w, h = cv2.boundingRect(best)
        
        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, 0.02 * peri, True)
        if len(approx) == 4:
            src = self._order_points(np.float32([pt[0] for pt in approx]))
            dst = np.float32([[0,0], [w,0], [w,h], [0,h]])
            H, _ = cv2.findHomography(src, dst)
            if H is not None:
                warp = cv2.warpPerspective(bgr_image, H, (w, h))
                v_pad, h_pad = int(h * 0.02), int(w * 0.02)
                return warp[v_pad:h-v_pad, h_pad:w-h_pad]
        
        return bgr_image[y:y+h, x:x+w]

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s, d = pts.sum(axis=1), np.diff(pts, axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
        return rect

    def publish_clue(self, loc, prediction):
        msg = f"{self.team_id},{self.password},{loc},{prediction}"
        self.score_pub.publish(String(data=msg))
        rospy.loginfo(f"PUBLISHED: {msg}")

    def run(self): rospy.spin()

if __name__ == '__main__':
    ClueReaderNode().run()