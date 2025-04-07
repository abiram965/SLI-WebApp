from io import BytesIO
import math
import os
import numpy as np
import tensorflow as tf
import base64
import time
import logging
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration constants
DATA_PATH = "dataset"
MODEL_PATH = "models/sign_language_model.h5"
IMG_SIZE = 224
OFFSET = 20
MAX_IMAGES_PER_LABEL = 100
MAX_HANDS = 2

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Initialize
detector = HandDetector(maxHands=MAX_HANDS)
capture_counts = {}

# Load existing model if available
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        labels = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
        num_classes = model.output_shape[-1]  # Get number of classes from model
        if len(labels) != num_classes:
            logger.warning(f"Labels ({len(labels)}) do not match model classes ({num_classes}).")
            # You might want to raise an error or adjust labels here depending on your needs
        logger.info(f"Model loaded successfully. Labels: {labels}")
    else:
        model = None
        labels = []
        logger.info("No pre-existing model found.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    labels = []

def preprocess_hand_image(img, hand):
    """Extract and preprocess a hand image for training/prediction."""
    try:
        x, y, w, h = hand['bbox']
        img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        h_img, w_img, _ = img.shape
        y1, y2 = max(0, y - OFFSET), min(h_img, y + h + OFFSET)
        x1, x2 = max(0, x - OFFSET), min(w_img, x + w + OFFSET)
        img_crop = img[y1:y2, x1:x2]

        if img_crop.size == 0:
            return None

        aspect_ratio = h / w
        if aspect_ratio > 1:
            k = IMG_SIZE / h
            w_cal = int(k * w)
            img_resize = cv2.resize(img_crop, (w_cal, IMG_SIZE))
            w_gap = (IMG_SIZE - w_cal) // 2
            img_white[:, w_gap:w_gap + w_cal] = img_resize
        else:
            k = IMG_SIZE / w
            h_cal = int(k * h)
            img_resize = cv2.resize(img_crop, (IMG_SIZE, h_cal))
            h_gap = (IMG_SIZE - h_cal) // 2
            img_white[h_gap:h_gap + h_cal, :] = img_resize

        return img_white
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def process_hand(img, hand, label, folder):
    """Process and save detected hand sign images."""
    try:
        img_white = preprocess_hand_image(img, hand)
        if img_white is None:
            return False

        label_folder = os.path.join(folder, label)
        os.makedirs(label_folder, exist_ok=True)

        existing_images = len(os.listdir(label_folder))
        if existing_images >= MAX_IMAGES_PER_LABEL:
            return False

        filename = os.path.join(label_folder, f'img_{int(time.time() * 1000)}.jpg')
        success = cv2.imwrite(filename, img_white)
        if not success:
            logger.error(f"Failed to save image: {filename}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error processing hand: {str(e)}")
        return False

@app.route('/capture_sign', methods=['POST'])
def capture_sign():
    """Receive image data and save hand sign images."""
    try:
        data = request.get_json()
        if not data or 'label' not in data or 'image' not in data:
            return jsonify({'error': 'Missing label or image data'}), 400

        label = data['label']
        img_data = data['image']

        label = "".join(c for c in label if c.isalnum() or c in ['-', '_']).strip()
        if not label:
            return jsonify({'error': 'Invalid label'}), 400

        if ',' in img_data:
            img_data = img_data.split(',')[1]

        np_arr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None or img.size == 0:
            return jsonify({'error': 'Invalid image data'}), 400

        hands, _ = detector.findHands(img, draw=False)
        saved_count = 0
        if hands:
            for hand in hands[:MAX_HANDS]:
                if process_hand(img, hand, label, DATA_PATH):
                    saved_count += 1
                    capture_counts[label] = capture_counts.get(label, 0) + 1

        return jsonify({
            'message': f'Processed {saved_count} hand(s)',
            'count': capture_counts.get(label, 0)
        })
    except Exception as e:
        logger.error(f"Error in capture_sign: {str(e)}")
        return jsonify({'error': f'Failed to process request: {str(e)}'}), 500

def load_image_data(directory):
    """Load images and labels from the dataset directory."""
    try:
        images, labels = [], []
        label_map = {}
        label_counter = 0

        subdirs = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
        if not subdirs:
            logger.warning("No label directories found")
            return None, None, {}

        logger.info(f"Found {len(subdirs)} label categories")

        for label in subdirs:
            label_path = os.path.join(directory, label)
            if label not in label_map:
                label_map[label] = label_counter
                label_counter += 1

            image_files = os.listdir(label_path)
            if not image_files:
                logger.warning(f"No images found for label '{label}'")
                continue

            logger.info(f"Loading {len(image_files)} images for label '{label}'")
            for image_file in image_files:
                image_path = os.path.join(label_path, image_file)
                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"Could not read image: {image_path}")
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label_map[label])

        if not images:
            logger.warning("No valid images found in any category")
            return None, None, {}

        logger.info(f"Successfully loaded {len(images)} images from {len(label_map)} categories")
        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), label_map
    except Exception as e:
        logger.error(f"Error loading image data: {str(e)}")
        return None, None, {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/capture_sign', methods=['GET'])
def capture_sign_page():
    return render_template('capture_sign.html')

@app.route('/train_model', methods=['GET'])
def train_model_page():
    return render_template('train_model.html')

@app.route('/detect_sign', methods=['GET'])
def detect_sign():
    return render_template('detect_sign.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train the sign language recognition model."""
    global model, labels

    try:
        X, y, label_map = load_image_data(DATA_PATH)
        if X is None or y is None or len(label_map) == 0:
            return jsonify({"error": "No valid training data available"}), 400

        if len(label_map) < 2:
            return jsonify({"error": "Need at least 2 different sign categories for training"}), 400

        labels = sorted(label_map.keys())
        num_classes = len(label_map)
        if num_classes != len(labels):
            logger.error(f"Number of classes ({num_classes}) does not match labels ({len(labels)})")
            return jsonify({"error": "Label mismatch during training"}), 500

        X = X / 255.0
        y = to_categorical(y, num_classes=num_classes)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Add data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train with data augmentation
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=15,
            validation_data=(X_test, y_test),
            verbose=1
        )

        loss, accuracy = model.evaluate(X_test, y_test)
        model.save(MODEL_PATH)

        return jsonify({
            "message": "Model training completed!",
            "accuracy": float(accuracy),
            "classes": labels
        })
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return jsonify({"error": f"Error during training: {str(e)}"}), 500

@app.route('/predict_sign', methods=['POST'])
def predict_sign():
    """Predict the sign in an uploaded image."""
    global model, labels

    if model is None:
        return jsonify({"error": "Model not trained yet. Please train the model first."}), 400

    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data received"}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
        image = np.array(image)

        hands, img = detector.findHands(image, draw=False)
        if not hands:
            return jsonify({"prediction": "No hand detected", "confidence": 0})

        hand = hands[0]
        img_white = preprocess_hand_image(img, hand)
        if img_white is None:
            return jsonify({"prediction": "Invalid hand image", "confidence": 0})

        img_white = np.expand_dims(img_white, axis=0) / 255.0
        prediction = model.predict(img_white, verbose=0)
        logger.debug(f"Prediction shape: {prediction[0].shape}, Prediction: {prediction[0]}")
        logger.debug(f"Labels length: {len(labels)}, Labels: {labels}")

        predicted_index = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_index])
        if predicted_index >= len(labels):
            logger.warning(f"Predicted index {predicted_index} out of bounds for labels size {len(labels)}")
            predicted_label = "Unknown"
        else:
            predicted_label = labels[predicted_index]

        # Ensure all_predictions only includes valid indices
        all_predictions = {
            labels[i]: float(prediction[0][i]) 
            for i in range(min(len(labels), len(prediction[0])))
        }

        return jsonify({
            "prediction": predicted_label,
            "confidence": confidence,
            "all_predictions": all_predictions
        })
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": f"Error in prediction: {str(e)}"}), 500

@app.route('/get_labels', methods=['GET'])
def get_labels():
    """Return the currently available labels."""
    try:
        available_labels = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
        return jsonify({"labels": available_labels})
    except Exception as e:
        logger.error(f"Error fetching labels: {str(e)}")
        return jsonify({"error": f"Error fetching labels: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Return the current status of the model and dataset."""
    try:
        label_counts = {}
        total_images = 0
        if os.path.exists(DATA_PATH):
            for label in os.listdir(DATA_PATH):
                label_path = os.path.join(DATA_PATH, label)
                if os.path.isdir(label_path):
                    count = len(os.listdir(label_path))
                    label_counts[label] = count
                    total_images += count

        return jsonify({
            "model_loaded": model is not None,
            "available_labels": sorted(label_counts.keys()),
            "total_images": total_images,
            "label_counts": label_counts,
            "model_path": MODEL_PATH if model is not None else None
        })
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return jsonify({"error": f"Error getting status: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)