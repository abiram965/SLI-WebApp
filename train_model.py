from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import threading

app = Flask(__name__)

TRAINING_STATUS = {'status': 'idle', 'message': 'Waiting to start training.'}

# Function to train the model
def train_model():
    global TRAINING_STATUS
    TRAINING_STATUS['status'] = 'training'
    TRAINING_STATUS['message'] = 'Training in progress...'

    data_dir = 'data'  # Folder containing captured hand sign images
    img_size = 224
    batch_size = 16

    try:
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        val_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(img_size, img_size),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(len(train_generator.class_indices), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_generator, validation_data=val_generator, epochs=10)

        model.save('hand_sign_model.h5')
        TRAINING_STATUS['status'] = 'completed'
        TRAINING_STATUS['message'] = 'Training complete! Model saved as hand_sign_model.h5.'
    
    except Exception as e:
        TRAINING_STATUS['status'] = 'error'
        TRAINING_STATUS['message'] = f'Training failed: {str(e)}'

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Train Model Page
@app.route('/train')
def train_page():
    return render_template('train.html')

# Start Model Training
@app.route('/train_model', methods=['POST'])
def start_training():
    if TRAINING_STATUS['status'] == 'training':
        return jsonify({'status': 'error', 'message': 'Training already in progress.'})

    thread = threading.Thread(target=train_model)
    thread.start()
    return jsonify({'status': 'started', 'message': 'Training started successfully.'})

# Get Training Status
@app.route('/training_status')
def training_status():
    return jsonify(TRAINING_STATUS)

if __name__ == '__main__':
    app.run(debug=True)
