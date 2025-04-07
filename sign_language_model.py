import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import logging

class SignLanguageModel:
    def __init__(self, data_path='data', model_path='sign_language_model.h5', img_size=224):
        """
        Initialize Sign Language Model
        
        Args:
            data_path (str): Directory containing sign language images
            model_path (str): Path to save/load model
            img_size (int): Input image size for model
        """
        self.data_path = data_path
        self.model_path = model_path
        self.img_size = img_size
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

    def load_image_data(self):
        """
        Load and preprocess image data
        
        Returns:
            tuple: Images, labels, and label mapping
        """
        images = []
        labels = []
        label_map = {}
        label_counter = 0

        for label in sorted(os.listdir(self.data_path)):
            label_path = os.path.join(self.data_path, label)
            if not os.path.isdir(label_path):
                continue

            if label not in label_map:
                label_map[label] = label_counter
                label_counter += 1

            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                img = cv2.imread(image_path)
                
                if img is None:
                    self.logger.warning(f"Unable to read image {image_path}")
                    continue
                
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                images.append(img)
                labels.append(label_map[label])

        images = np.array(images)
        labels = np.array(labels)
        return images, labels, label_map

    def create_model(self, num_classes):
        """
        Create CNN model for sign language classification
        
        Args:
            num_classes (int): Number of sign language classes
            
        Returns:
            keras.Model: Compiled neural network model
        """
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Flatten and dense layers
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        return model

    def train_model(self):
        """
        Train the sign language classification model
        
        Returns:
            tuple: Trained model and label map
        """
        # Load data
        X, y, label_map = self.load_image_data()
        
        if len(X) == 0:
            raise ValueError("No training data found. Please capture some images first.")
        
        y_categorical = to_categorical(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42
        )
        
        # Normalize pixel values
        X_train, X_test = X_train / 255.0, X_test / 255.0

        # Create model
        model = self.create_model(len(label_map))

        # Callbacks
        checkpoint = ModelCheckpoint(
            self.model_path, 
            monitor='val_accuracy', 
            save_best_only=True
        )
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )

        # Train model
        history = model.fit(
            X_train, y_train, 
            epochs=50, 
            validation_data=(X_test, y_test),
            callbacks=[checkpoint, early_stop],
            batch_size=32
        )

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        self.logger.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        return model, label_map

    def load_trained_model(self):
        """
        Load pre-trained model
        
        Returns:
            tuple: Loaded model and label map
        """
        try:
            model = keras.models.load_model(self.model_path)
            
            # Recreate label map
            label_map = {label: idx for idx, label in enumerate(
                sorted(os.listdir(self.data_path))
            )}
            
            return model, label_map
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None, None

def main():
    # Example usage
    sign_model = SignLanguageModel()
    
    try:
        model, label_map = sign_model.train_model()
        print("Labels:", list(label_map.keys()))
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()