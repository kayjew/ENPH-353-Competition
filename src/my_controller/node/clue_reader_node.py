#!/usr/bin/env python3
import os
import sys
import rospy
import rospkg
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import tensorflow as tf
from std_msgs.msg import Bool, String, Float32

rospack = rospkg.RosPack()
# Get the path to my_controller package
pkg_path = rospack.get_path('my_controller')
MODEL_PATH = os.path.join(pkg_path, 'models', 'clue_reader_local.h5')
SHOW_DEBUG_VIEW = True
TEAM_ID         = 'TeamName'
PASSWORD        = 'password'
CAMERA_TOPIC    = 'B1/rrbot/camera1/image_raw'
PROCESS_EVERY_N = 1
CONFIRM_COUNT   = 3 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from clue_detection.model_utils import BoardProcessor, int_to_char

class ClueReaderNode:
    def __init__(self):
        rospy.init_node('clue_reader_node', anonymous=True)
        self.offset_pub = rospy.Publisher("/clue/horizontal_offset", Float32, queue_size=1)

        self.team_id         = TEAM_ID
        self.password        = PASSWORD
        self.process_every_n = PROCESS_EVERY_N
        self.frame_count     = 0
        self.last_board_x = 250

        self.last_val           = None
        self.last_type_id       = None
        self.consecutive_count  = 0
        self.published_clue_ids = set()

        self.last_match_key = ""
        self.consecutive_count = 0
        self.published_ids = set() # To keep track of which boards we've already scored
        
        # Map the fuzzy ID to the competition board number (example mapping)
        self.id_map = {
            "SIZE": 1,
            "VICTIM": 2,
            "CRIME": 3,
            "TIME": 4,
            "PLACE": 5,
            "MOTIVE": 6,
            "WEAPON": 7,
            "BANDIT": 8,
        }

        self.model = tf.keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        if not self.model: 
            rospy.logerr("Model not found!")

        self.bridge    = CvBridge()
        self.processor = BoardProcessor()
        self.score_pub = rospy.Publisher('/score_tracker', String, queue_size=10)

        self.processing_pub = rospy.Publisher("/clue/active_processing", Bool, queue_size=1)

        rospy.Subscriber(CAMERA_TOPIC, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo("[ClueReader] Clue finding active")

    def get_fuzzy_id(self, raw_id):
        raw_id = raw_id.upper()
        
        # Define keywords
        mapping = {
            "SIZE": ["SIZE", "S1ZE", "5IZE"],
            "VICTIM": ["VIC", "V1C", "TIM", "V1CT", "VIT"],
            "CRIME": ["CRIM", "CRME", "CR1M", "RIME", "IHE", "1HE"],
            "PLACE": ["PLAC", "PLCE", "LACE"],
        }
        
        for official_name, triggers in mapping.items():
            if any(t in raw_id for t in triggers):
                return official_name
        return None # No clear match
    
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

        if result is None:
            self.processing_pub.publish(False)
            self.consecutive_count = 0
            return

        raw_type, raw_val = result
        fuzzy_type = self.get_fuzzy_id(raw_type)
        board_num = self.id_map.get(fuzzy_type, 0)

        if board_num in self.published_ids:
            self.processing_pub.publish(False) 
            return 

        self.processing_pub.publish(True)
        img_w = cv_img.shape[1]
        normalized_offset = (self.last_board_x - (img_w / 2)) / (img_w / 2)
        self.offset_pub.publish(normalized_offset)

        clean_val = raw_val.replace(" ", "").strip()
        if fuzzy_type and clean_val:
            match_key = f"{fuzzy_type}:{clean_val}"
            
            if match_key == self.last_match_key:
                self.consecutive_count += 1
            else:
                self.consecutive_count = 1
                self.last_match_key = match_key

            if self.consecutive_count >= 3:
                submission = f"TeamName,password,{board_num},{clean_val}"
                self.score_pub.publish(submission)
                self.published_ids.add(board_num)
                rospy.loginfo(f"PUBLISHED: {submission}")
                
                self.consecutive_count = 0 
                self.last_match_key = ""
                self.processing_pub.publish(False)

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
            if area < 3000: continue
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
        self.last_board_x = x + (w/2)
        
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
        self.processing_pub.publish(False)
        rospy.loginfo(f"PUBLISHED: {msg}")

    def run(self): rospy.spin()

if __name__ == '__main__':
    ClueReaderNode().run()