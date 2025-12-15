#!/usr/bin/env python3
"""
AI Model Training Script for Cheat Detection
Supports OpenCV, MediaPipe, TensorFlow, PyTorch, and YOLOv8
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
from ultralytics import YOLO
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any

class CheatDetectionDataset(Dataset):
    """Custom dataset for cheat detection training"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label

class CheatDetectionNN(nn.Module):
    """Neural Network for cheat detection"""
    
    def __init__(self, input_size: int = 15, hidden_sizes: List[int] = [64, 32, 16]):
        super(CheatDetectionNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class MediaPipeFeatureExtractor:
    """Extract features using MediaPipe"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract comprehensive features from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        face_results = self.face_mesh.process(rgb_image)
        hand_results = self.hands.process(rgb_image)
        pose_results = self.pose.process(rgb_image)
        
        features = []
        
        # Face features
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # Gaze direction estimation
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]
            
            gaze_x = (left_eye.x + right_eye.x) / 2 - nose_tip.x
            gaze_y = (left_eye.y + right_eye.y) / 2 - nose_tip.y
            
            # Head pose estimation
            left_ear = face_landmarks.landmark[234]
            right_ear = face_landmarks.landmark[454]
            chin = face_landmarks.landmark[175]
            forehead = face_landmarks.landmark[10]
            
            yaw = np.arctan2(right_ear.x - left_ear.x, right_ear.z - left_ear.z) * 180 / np.pi
            pitch = np.arctan2(forehead.y - chin.y, forehead.z - chin.z) * 180 / np.pi
            roll = np.arctan2(right_ear.y - left_ear.y, right_ear.x - left_ear.x) * 180 / np.pi
            
            # Eye aspect ratio (blink detection)
            left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            left_ear = self.calculate_ear(face_landmarks, left_eye_landmarks)
            right_ear = self.calculate_ear(face_landmarks, right_eye_landmarks)
            
            features.extend([gaze_x, gaze_y, yaw, pitch, roll, left_ear, right_ear, 1])  # face_detected = 1
        else:
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])  # face_detected = 0
        
        # Hand features
        if hand_results.multi_hand_landmarks:
            num_hands = len(hand_results.multi_hand_landmarks)
            
            # Hand position relative to face
            if num_hands > 0:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                wrist = hand_landmarks.landmark[0]
                features.extend([wrist.x, wrist.y, num_hands])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])
        
        # Pose features
        if pose_results.pose_landmarks:
            # Shoulder and head positions
            left_shoulder = pose_results.pose_landmarks.landmark[11]
            right_shoulder = pose_results.pose_landmarks.landmark[12]
            nose = pose_results.pose_landmarks.landmark[0]
            
            shoulder_angle = np.arctan2(right_shoulder.y - left_shoulder.y, 
                                      right_shoulder.x - left_shoulder.x) * 180 / np.pi
            
            features.extend([shoulder_angle, nose.x, nose.y, 1])  # pose_detected = 1
        else:
            features.extend([0, 0, 0, 0])  # pose_detected = 0
        
        return np.array(features, dtype=np.float32)
    
    def calculate_ear(self, landmarks, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        # Get eye landmarks
        eye_points = []
        for idx in eye_landmarks[:6]:  # Use first 6 landmarks for EAR calculation
            point = landmarks.landmark[idx]
            eye_points.append([point.x, point.y])
        
        eye_points = np.array(eye_points)
        
        # Calculate EAR
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        ear = (A + B) / (2.0 * C)
        return ear

class YOLOCheatDetector:
    """YOLO-based object detection for cheat detection"""
    
    def __init__(self, model_path: str = None):
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')  # Use pre-trained YOLOv8 nano
    
    def detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect objects that might indicate cheating"""
        results = self.model(image)
        
        detections = {
            'phones': 0,
            'books': 0,
            'persons': 0,
            'laptops': 0,
            'confidence_scores': []
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.model.names[class_id]
                    
                    detections['confidence_scores'].append(confidence)
                    
                    if class_name == 'cell phone':
                        detections['phones'] += 1
                    elif class_name == 'book':
                        detections['books'] += 1
                    elif class_name == 'person':
                        detections['persons'] += 1
                    elif class_name == 'laptop':
                        detections['laptops'] += 1
        
        return detections

class CheatDetectionTrainer:
    """Main trainer class for cheat detection model"""
    
    def __init__(self, data_path: str = "training_data"):
        self.data_path = data_path
        self.feature_extractor = MediaPipeFeatureExtractor()
        self.yolo_detector = YOLOCheatDetector()
        self.scaler = StandardScaler()
        
        # Create directories
        os.makedirs(data_path, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def collect_training_data(self, video_path: str, labels_path: str):
        """Collect training data from video with labels"""
        cap = cv2.VideoCapture(video_path)
        
        # Load labels (timestamp -> label mapping)
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        features_list = []
        labels_list = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)
            
            # Extract features
            mp_features = self.feature_extractor.extract_features(frame)
            yolo_detections = self.yolo_detector.detect_objects(frame)
            
            # Combine features
            combined_features = np.concatenate([
                mp_features,
                [yolo_detections['phones'], yolo_detections['books'], 
                 yolo_detections['persons'], yolo_detections['laptops']]
            ])
            
            # Get label for this timestamp
            label = self.get_label_for_timestamp(timestamp, labels)
            
            features_list.append(combined_features)
            labels_list.append(label)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        
        return np.array(features_list), np.array(labels_list)
    
    def get_label_for_timestamp(self, timestamp: float, labels: Dict) -> int:
        """Get label for given timestamp"""
        for label_info in labels:
            start_time = label_info['start_time']
            end_time = label_info['end_time']
            if start_time <= timestamp <= end_time:
                return 1 if label_info['is_cheating'] else 0
        return 0  # Default to not cheating
    
    def train_pytorch_model(self, X: np.ndarray, y: np.ndarray):
        """Train PyTorch model"""
        print("Training PyTorch model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create datasets
        train_dataset = CheatDetectionDataset(X_train_scaled, y_train)
        test_dataset = CheatDetectionDataset(X_test_scaled, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = CheatDetectionNN(input_size=X.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        train_losses = []
        test_losses = []
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Testing
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for features, labels in test_loader:
                    outputs = model(features).squeeze()
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    
                    predicted = (outputs > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)
            accuracy = correct / total
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            scheduler.step(test_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, '
                      f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': self.scaler,
            'input_size': X.shape[1]
        }, 'models/pytorch_cheat_detection.pth')
        
        return model, train_losses, test_losses
    
    def train_tensorflow_model(self, X: np.ndarray, y: np.ndarray):
        """Train TensorFlow model"""
        print("Training TensorFlow model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=10),
            tf.keras.callbacks.ModelCheckpoint('models/tensorflow_cheat_detection.h5', 
                                             save_best_only=True)
        ]
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Convert to TensorFlow.js format
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, 'models/tfjs_model')
        
        return model, history
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, model_type: str):
        """Evaluate trained model"""
        print(f"Evaluating {model_type} model...")
        
        if model_type == 'pytorch':
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(self.scaler.transform(X_test))
                predictions = model(X_test_tensor).squeeze().numpy()
        else:  # tensorflow
            predictions = model.predict(self.scaler.transform(X_test)).squeeze()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        binary_predictions = (predictions > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, binary_predictions)
        precision = precision_score(y_test, binary_predictions)
        recall = recall_score(y_test, binary_predictions)
        f1 = f1_score(y_test, binary_predictions)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, binary_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_type.capitalize()} Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'logs/{model_type}_confusion_matrix.png')
        plt.close()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def create_sample_data(self, num_samples: int = 1000):
        """Create sample synthetic data for demonstration"""
        print(f"Creating {num_samples} synthetic samples...")
        
        np.random.seed(42)
        
        # Generate synthetic features
        features = []
        labels = []
        
        for i in range(num_samples):
            # Normal behavior (60% of data)
            if i < num_samples * 0.6:
                # Normal gaze direction (looking at screen)
                gaze_x = np.random.normal(0, 0.1)
                gaze_y = np.random.normal(0, 0.1)
                
                # Normal head pose
                yaw = np.random.normal(0, 10)
                pitch = np.random.normal(0, 10)
                roll = np.random.normal(0, 5)
                
                # Normal eye aspect ratio
                left_ear = np.random.normal(0.3, 0.05)
                right_ear = np.random.normal(0.3, 0.05)
                
                # Face detected
                face_detected = 1
                
                # Normal hand position
                hand_x = np.random.normal(0.5, 0.2)
                hand_y = np.random.normal(0.7, 0.1)
                num_hands = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
                
                # Normal pose
                shoulder_angle = np.random.normal(0, 5)
                nose_x = np.random.normal(0.5, 0.1)
                nose_y = np.random.normal(0.3, 0.1)
                pose_detected = 1
                
                # No suspicious objects
                phones = 0
                books = 0
                persons = 1
                laptops = np.random.choice([0, 1], p=[0.7, 0.3])
                
                label = 0  # Not cheating
            
            # Suspicious behavior (40% of data)
            else:
                # Suspicious gaze (looking away)
                gaze_x = np.random.normal(0, 0.4)
                gaze_y = np.random.normal(0, 0.4)
                
                # Suspicious head pose
                yaw = np.random.normal(0, 30)
                pitch = np.random.normal(0, 25)
                roll = np.random.normal(0, 15)
                
                # Varied eye aspect ratio (might be looking away)
                left_ear = np.random.normal(0.25, 0.1)
                right_ear = np.random.normal(0.25, 0.1)
                
                # Sometimes face not detected
                face_detected = np.random.choice([0, 1], p=[0.3, 0.7])
                
                # Suspicious hand movements
                hand_x = np.random.normal(0.5, 0.3)
                hand_y = np.random.normal(0.5, 0.3)
                num_hands = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
                
                # Suspicious pose
                shoulder_angle = np.random.normal(0, 20)
                nose_x = np.random.normal(0.5, 0.2)
                nose_y = np.random.normal(0.4, 0.2)
                pose_detected = np.random.choice([0, 1], p=[0.2, 0.8])
                
                # Suspicious objects
                phones = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
                books = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
                persons = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                laptops = np.random.choice([0, 1, 2], p=[0.5, 0.4, 0.1])
                
                label = 1  # Cheating
            
            feature_vector = [
                gaze_x, gaze_y, yaw, pitch, roll, left_ear, right_ear, face_detected,
                hand_x, hand_y, num_hands, shoulder_angle, nose_x, nose_y, pose_detected,
                phones, books, persons, laptops
            ]
            
            features.append(feature_vector)
            labels.append(label)
        
        return np.array(features), np.array(labels)

def main():
    """Main training function"""
    trainer = CheatDetectionTrainer()
    
    # For demonstration, create synthetic data
    # In production, use real video data with labels
    print("Creating synthetic training data...")
    X, y = trainer.create_sample_data(num_samples=5000)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Positive samples: {np.sum(y)}")
    print(f"Negative samples: {len(y) - np.sum(y)}")
    
    # Train PyTorch model
    pytorch_model, train_losses, test_losses = trainer.train_pytorch_model(X, y)
    
    # Train TensorFlow model
    tf_model, history = trainer.train_tensorflow_model(X, y)
    
    # Evaluate models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pytorch_metrics = trainer.evaluate_model(pytorch_model, X_test, y_test, 'pytorch')
    tf_metrics = trainer.evaluate_model(tf_model, X_test, y_test, 'tensorflow')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('PyTorch Model - Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('TensorFlow Model - Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('logs/training_history.png')
    plt.close()
    
    # Save training report
    report = {
        'pytorch_metrics': pytorch_metrics,
        'tensorflow_metrics': tf_metrics,
        'dataset_info': {
            'total_samples': len(X),
            'features': X.shape[1],
            'positive_samples': int(np.sum(y)),
            'negative_samples': int(len(y) - np.sum(y))
        }
    }
    
    with open('logs/training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nTraining completed!")
    print("Models saved in 'models/' directory")
    print("Logs and visualizations saved in 'logs/' directory")
    print("\nModel Performance Summary:")
    print(f"PyTorch Model - Accuracy: {pytorch_metrics['accuracy']:.4f}, F1: {pytorch_metrics['f1_score']:.4f}")
    print(f"TensorFlow Model - Accuracy: {tf_metrics['accuracy']:.4f}, F1: {tf_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()