# AI-Powered Cheat Detection Training

This directory contains the training scripts and tools for developing AI models to detect cheating behavior during online exams.

## Features

- **Multi-Modal AI Detection**: Combines OpenCV, MediaPipe, TensorFlow, PyTorch, and YOLOv8
- **Real-time Processing**: Optimized for live webcam monitoring
- **Comprehensive Feature Extraction**: Gaze tracking, head pose estimation, hand detection, object recognition
- **Multiple Model Architectures**: Both PyTorch and TensorFlow implementations
- **Data Collection Tools**: GUI-based tool for collecting labeled training data

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU support (recommended):
```bash
# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow[and-cuda]
```

## Data Collection

### Using the GUI Tool

Run the data collection GUI to create labeled training data:

```bash
python data_collection.py
```

The GUI allows you to:
- Record video from webcam
- Label behaviors in real-time
- Export labeled datasets for training

### Manual Data Preparation

If you have existing video data:

1. Place videos in `training_data/videos/`
2. Create corresponding label files in JSON format
3. Use the sample format provided in `training_data/sample/`

## Training Models

### Quick Start with Synthetic Data

```bash
python train_cheat_detection.py
```

This will:
- Generate synthetic training data
- Train both PyTorch and TensorFlow models
- Evaluate model performance
- Save models for deployment

### Training with Real Data

```bash
python train_cheat_detection.py --data_path /path/to/your/data --video_path /path/to/video.mp4 --labels_path /path/to/labels.json
```

## Model Architecture

### Features Extracted (19 dimensions):

1. **Gaze Direction** (2D): X, Y coordinates of estimated gaze
2. **Head Pose** (3D): Yaw, pitch, roll angles
3. **Eye Aspect Ratio** (2D): Left and right eye openness
4. **Face Detection** (1D): Binary face presence
5. **Hand Position** (3D): X, Y coordinates and count
6. **Body Pose** (4D): Shoulder angle, nose position, pose detection
7. **Object Detection** (4D): Phones, books, people, laptops count

### Neural Network Architecture:

```
Input (19) → Dense(64) → Dropout(0.3) → BatchNorm → 
Dense(32) → Dropout(0.3) → BatchNorm → 
Dense(16) → Dropout(0.3) → 
Dense(1) → Sigmoid
```

## Suspicious Behaviors Detected

- **Gaze Deviation**: Looking away from screen
- **Head Movement**: Unusual head poses
- **Face Absence**: No face detected in frame
- **Multiple People**: More than one person detected
- **Prohibited Objects**: Phones, books, additional devices
- **Hand Gestures**: Suspicious hand movements
- **Audio Anomalies**: Unusual sound patterns

## Model Performance

Expected performance on balanced dataset:
- **Accuracy**: 85-95%
- **Precision**: 80-90%
- **Recall**: 85-95%
- **F1-Score**: 82-92%

## Deployment

### TensorFlow.js Model

The trained TensorFlow model is automatically converted to TensorFlow.js format for browser deployment:

```javascript
// Load model in browser
const model = await tf.loadLayersModel('/models/tfjs_model/model.json');

// Make predictions
const features = tf.tensor2d([[gaze_x, gaze_y, ...]]); // 19 features
const prediction = model.predict(features);
```

### PyTorch Model

For server-side deployment:

```python
# Load model
checkpoint = torch.load('models/pytorch_cheat_detection.pth')
model = CheatDetectionNN(input_size=19)
model.load_state_dict(checkpoint['model_state_dict'])
scaler = checkpoint['scaler']

# Make predictions
features_scaled = scaler.transform(features)
prediction = model(torch.FloatTensor(features_scaled))
```

## File Structure

```
training/
├── train_cheat_detection.py    # Main training script
├── data_collection.py          # GUI data collection tool
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/                     # Saved models
│   ├── pytorch_cheat_detection.pth
│   ├── tensorflow_cheat_detection.h5
│   └── tfjs_model/            # TensorFlow.js model
├── logs/                      # Training logs and visualizations
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── training_report.json
└── training_data/             # Training datasets
    ├── videos/
    ├── labels/
    └── sample/
```

## Advanced Usage

### Custom Feature Engineering

Modify the `MediaPipeFeatureExtractor` class to add new features:

```python
def extract_custom_features(self, image):
    # Add your custom feature extraction logic
    pass
```

### Hyperparameter Tuning

Adjust model parameters in the training script:

```python
# Neural network architecture
hidden_sizes = [128, 64, 32]  # Increase model capacity
dropout_rate = 0.4            # Adjust regularization
learning_rate = 0.0005        # Fine-tune learning rate
```

### Real-time Optimization

For production deployment, consider:
- Model quantization for faster inference
- Feature caching to reduce computation
- Batch processing for multiple students
- GPU acceleration for real-time processing

## Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera permissions and drivers
2. **CUDA out of memory**: Reduce batch size or use CPU training
3. **Poor model performance**: Collect more diverse training data
4. **Slow inference**: Enable GPU acceleration or model optimization

### Performance Optimization

- Use GPU acceleration: `CUDA_VISIBLE_DEVICES=0 python train_cheat_detection.py`
- Reduce model complexity for real-time inference
- Implement feature caching for repeated computations
- Use model quantization for deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Ensure compliance with privacy laws and institutional policies when deploying in production environments.