import cv2
import os
import time
from cvzone.HandTrackingModule import HandDetector

class HandSignCapture:
    def __init__(self, max_hands=1, img_size=224, offset=20, output_folder="data"):
        self.max_hands = max_hands
        self.img_size = img_size
        self.offset = offset
        self.output_folder = output_folder
        self.detector = HandDetector(maxHands=self.max_hands)

    def process_frame(self, img, label, counter, save_limit):
        """Detects hands, crops, resizes, and saves hand images."""
        hands, img = self.detector.findHands(img, draw=True)

        if hands and label:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Crop and process the hand region
            img_crop = img[max(0, y - self.offset): y + h + self.offset,
                           max(0, x - self.offset): x + w + self.offset]
            if img_crop.size == 0:
                return img, counter  # Skip empty frames

            img_resized = cv2.resize(img_crop, (self.img_size, self.img_size))

            # Save image
            folder_path = os.path.join(self.output_folder, label)
            os.makedirs(folder_path, exist_ok=True)

            if counter < save_limit:
                filename = os.path.join(folder_path, f'Image_{time.time()}.jpg')
                cv2.imwrite(filename, img_resized)
                counter += 1

        return img, counter
