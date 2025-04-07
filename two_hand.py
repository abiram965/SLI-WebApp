import os
import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
import logging

class HandSignCapture:
    def __init__(self, max_hands=2, img_size=224, offset=20):
        """
        Initialize hand sign capture system
        
        Args:
            max_hands (int): Maximum number of hands to detect
            img_size (int): Size of output image
            offset (int): Padding around detected hand
        """
        self.max_hands = max_hands
        self.img_size = img_size
        self.offset = offset
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

    def process_hand(self, img, hand, label, counter, folder):
        """
        Process a single hand for image capture
        
        Args:
            img (numpy.ndarray): Original image
            hand (dict): Hand detection information
            label (str): Label for the hand sign
            counter (int): Current image counter
            folder (str): Folder to save images
        
        Returns:
            int: Updated counter
        """
        x, y, w, h = hand['bbox']
        
        # Create white background image
        img_white = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255
        
        # Compute crop coordinates with safety checks
        h_img, w_img, _ = img.shape
        y1, y2 = max(0, y - self.offset), min(h_img, y + h + self.offset)
        x1, x2 = max(0, x - self.offset), min(w_img, x + w + self.offset)
        img_crop = img[y1:y2, x1:x2]
        
        if img_crop.size == 0:
            return counter
        
        # Maintain aspect ratio during resize
        aspect_ratio = h / w
        if aspect_ratio > 1:
            k = self.img_size / h
            w_cal = math.ceil(k * w)
            img_resize = cv2.resize(img_crop, (w_cal, self.img_size))
            w_gap = math.ceil((self.img_size - w_cal) / 2)
            img_white[:, w_gap:w_gap + w_cal] = img_resize
        else:
            k = self.img_size / w
            h_cal = math.ceil(k * h)
            img_resize = cv2.resize(img_crop, (self.img_size, h_cal))
            h_gap = math.ceil((self.img_size - h_cal) / 2)
            img_white[h_gap:h_gap + h_cal, :] = img_resize
        
        # Show preview
        cv2.imshow(f"Hand Image {counter}", img_white)
        
        # Save image
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            filename = os.path.join(folder, f'Image_{time.time()}.jpg')
            cv2.imwrite(filename, img_white)
            self.logger.info(f"Saved {filename} - Total: {counter}")
        
        return counter

    def capture_signs(self, label, save_limit=100):
        """
        Capture hand sign images for a specific label
        
        Args:
            label (str): Label/category of hand sign
            save_limit (int): Maximum number of images to save
        """
        # Ensure data directory exists
        folder = os.path.join('dataset', label)
        os.makedirs(folder, exist_ok=True)
        
        # Initialize video capture and hand detector
        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=self.max_hands)
        
        counter = 0
        
        while counter < save_limit:
            success, img = cap.read()
            if not success:
                self.logger.error("Failed to grab frame")
                break

            # Find hands in the image
            hands, img_annotated = detector.findHands(img, draw=True)
            
            if hands:
                # Process each detected hand
                for hand_idx, hand in enumerate(hands):
                    # Update counter for each hand
                    counter = self.process_hand(img, hand, label, counter, folder)
                    
                    # Optional: Annotate hands with index
                    cv2.putText(img_annotated, 
                                f"Hand {hand_idx + 1}", 
                                (hand['bbox'][0], hand['bbox'][1] - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)
            
            # Show progress
            cv2.putText(img_annotated, 
                        f"Capturing {label}: {counter}/{save_limit}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)
            
            cv2.imshow("Hand Sign Capture", img_annotated)
            
            # Exit conditions
            if cv2.waitKey(1) & 0xFF == ord('q') or counter >= save_limit:
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        self.logger.info(f"Completed capturing {counter} images for {label}")

def main():
    # Example usage
    label = input("Enter label for hand sign: ").strip()
    hand_capture = HandSignCapture(max_hands=2)  # Set to 2 hands
    hand_capture.capture_signs(label)

if __name__ == "__main__":
    main()