import cv2
import numpy as np
import string

# Constants
CHARS = string.digits + string.ascii_uppercase
char_to_int = {char: i for i, char in enumerate(CHARS)}
int_to_char = {i: char for i, char in enumerate(CHARS)}


class BoardProcessor:
    def __init__(self, size=32):
        self.char_size = size
        # Coordinates for the ROI
        self.top_y1, self.top_y2, self.top_x1, self.top_x2 = 20,  130, 100, 490
        self.bot_y1, self.bot_y2, self.bot_x1, self.bot_x2 = 150, 274,  40, 490

    def resize_and_pad(self, img, size):
        h, w = img.shape
        if h < 2 or w < 2:
            return None
        scale = (size - 6) / max(h, w)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        square = np.zeros((size, size), dtype=np.uint8)
        offset_y = (size - new_h) // 2
        offset_x = (size - new_w) // 2
        square[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
        return square

    def get_chars_from_roi(self, roi):
        thresh = cv2.adaptiveThreshold(
            roi, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        # Remove small noise with open, then close small gaps
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 28 < h < 100 and 2 < w < 80:
                char_boxes.append((x, y, w, h))

        char_boxes.sort(key=lambda b: b[0])
        chars = []
        for x, y, w, h in char_boxes:
            img = self.resize_and_pad(thresh[y:y + h, x:x + w], self.char_size)
            if img is not None:
                chars.append(img)
        return chars

    def segment_both(self, gray_img):
        """
        takes in image, returns 2 strings for the clue type and value
        """
        type_chars = self.get_chars_from_roi(
            gray_img[self.top_y1:self.top_y2, self.top_x1:self.top_x2]
        )
        val_chars = self.get_chars_from_roi(
            gray_img[self.bot_y1:self.bot_y2, self.bot_x1:self.bot_x2]
        )
        return type_chars, val_chars