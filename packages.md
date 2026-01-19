# Python Packages for Computer Vision & Image Recognition

A curated list of Python libraries specifically designed for computer vision and object/person recognition tasks.

---

## 1. OpenCV (cv2)

**Use Cases:** General-purpose computer vision library for image processing, object detection, face detection, video analysis, and feature extraction. Widely used as a foundation for other CV tasks.

**Note:** OpenCV alone requires pre-trained Haar cascades or DNN models for person/gender detection. For gender classification, an external pre-trained model is needed.

```python
import cv2
import numpy as np

# 1. Initialize library with pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Gender model (requires downloading pre-trained caffemodel)
# Download from: https://github.com/GilLevi/AgeGenderDeepLearning
gender_net = cv2.dnn.readNetFromCaffe(
    'deploy_gender.prototxt',
    'gender_net.caffemodel'
)
GENDER_LIST = ['Male', 'Female']

# 2. Male/Female definitions (pre-trained model outputs)
# No training needed - uses pre-trained deep learning model
# Model outputs probability for [Male, Female] classes

# 3. Recognition method
def recognize_person_gender(image_path: str) -> tuple[bool, float, float]:
    """Returns (has_full_human, male_factor, female_factor)"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect full body
    bodies = body_cascade.detectMultiScale(gray, 1.1, 3)
    has_full_human = len(bodies) > 0

    male_factor, female_factor = 0.0, 0.0

    if has_full_human:
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = img[y:y+h, x:x+w]
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                         (78.4, 87.7, 114.9), swapRB=False)
            gender_net.setInput(blob)
            preds = gender_net.forward()
            # Scale predictions (0-1) to likeness factor (0-2)
            male_factor = float(preds[0][0]) * 2.0
            female_factor = float(preds[0][1]) * 2.0

    return has_full_human, male_factor, female_factor

# Usage
result = recognize_person_gender("test_image.jpg")
print(f"Human detected: {result[0]}, Male: {result[1]:.2f}, Female: {result[2]:.2f}")
```

---

## 2. MediaPipe

**Use Cases:** Google's framework for building perception pipelines. Excellent for pose estimation, face detection, hand tracking, and holistic body detection with real-time performance.

**Note:** MediaPipe provides excellent pose detection but does not include gender classification. A separate classifier is needed for gender.

```python
import mediapipe as mp
import cv2
import numpy as np

# 1. Initialize MediaPipe pose and face detection
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_detection

pose_detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
)
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

# 2. Gender definitions
# MediaPipe does not provide gender classification natively
# For simplicity, using pose landmark ratios as heuristic (not accurate for production)
# Production use requires: trained gender classifier on extracted features
SHOULDER_HIP_RATIO_MALE_THRESHOLD = 1.3  # Males typically have wider shoulders
SHOULDER_HIP_RATIO_FEMALE_THRESHOLD = 1.1

# 3. Recognition method
def recognize_person_gender(image_path: str) -> tuple[bool, float, float]:
    """Returns (has_full_human, male_factor, female_factor)"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pose_results = pose_detector.process(img_rgb)

    has_full_human = False
    male_factor, female_factor = 0.0, 0.0

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        # Check if key body points are visible (full body)
        visibility_threshold = 0.5
        key_points = [0, 11, 12, 23, 24, 27, 28]  # nose, shoulders, hips, ankles
        visible_count = sum(1 for i in key_points if landmarks[i].visibility > visibility_threshold)
        has_full_human = visible_count >= 5

        if has_full_human:
            # Calculate shoulder-hip ratio for gender heuristic
            shoulder_width = abs(landmarks[11].x - landmarks[12].x)
            hip_width = abs(landmarks[23].x - landmarks[24].x)
            ratio = shoulder_width / hip_width if hip_width > 0 else 1.0

            # Convert ratio to likeness factors (0-2 scale)
            male_factor = min(2.0, max(0.0, (ratio - 0.8) * 2.5))
            female_factor = min(2.0, max(0.0, (1.5 - ratio) * 2.5))

    return has_full_human, male_factor, female_factor

# Usage
result = recognize_person_gender("test_image.jpg")
print(f"Human detected: {result[0]}, Male: {result[1]:.2f}, Female: {result[2]:.2f}")
```

---

## 3. Ultralytics YOLO (YOLOv8)

**Use Cases:** State-of-the-art object detection with real-time performance. Excellent for detecting people, objects, and custom classes. Supports classification, segmentation, and pose estimation.

**Note:** YOLO detects persons but requires additional classification head or separate model for gender. Uses COCO pre-trained weights.

```python
from ultralytics import YOLO
import cv2
import numpy as np

# 1. Initialize YOLO models
person_detector = YOLO('yolov8n.pt')  # Pre-trained on COCO (includes 'person' class)
# For gender: requires custom trained model or separate classifier
# Simplified: using YOLOv8 classification with custom trained weights
# Training required: dataset of male/female full-body images
gender_classifier = YOLO('yolov8n-cls.pt')  # Would need fine-tuning for gender

# 2. Gender definitions
# Pre-trained COCO model: class 0 = 'person'
# For gender classification, you need:
# - Dataset: labeled male/female images (e.g., from CelebA, DeepFashion)
# - Training: gender_classifier.train(data='gender_dataset', epochs=50)
# Simplified placeholder for testing:
PERSON_CLASS_ID = 0
MALE_CLASS_ID = 0   # After custom training
FEMALE_CLASS_ID = 1  # After custom training

# 3. Recognition method
def recognize_person_gender(image_path: str) -> tuple[bool, float, float]:
    """Returns (has_full_human, male_factor, female_factor)"""
    img = cv2.imread(image_path)
    img_height = img.shape[0]

    # Detect persons
    results = person_detector(image_path, classes=[PERSON_CLASS_ID], verbose=False)

    has_full_human = False
    male_factor, female_factor = 0.0, 0.0

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == PERSON_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_height = y2 - y1
                # Check if person occupies significant vertical space (full body)
                if box_height > img_height * 0.5:
                    has_full_human = True
                    confidence = float(box.conf[0])

                    # Extract person crop for gender classification
                    person_crop = img[y1:y2, x1:x2]
                    gender_results = gender_classifier(person_crop, verbose=False)

                    # Get classification probabilities (after custom training)
                    if gender_results[0].probs is not None:
                        probs = gender_results[0].probs.data.numpy()
                        male_factor = float(probs[MALE_CLASS_ID]) * 2.0
                        female_factor = float(probs[FEMALE_CLASS_ID]) * 2.0
                    break

    return has_full_human, male_factor, female_factor

# Usage
result = recognize_person_gender("test_image.jpg")
print(f"Human detected: {result[0]}, Male: {result[1]:.2f}, Female: {result[2]:.2f}")
```

---

## 4. DeepFace

**Use Cases:** Lightweight face recognition and facial attribute analysis library. Built-in support for age, gender, emotion, and race detection using various backend models (VGG-Face, Facenet, OpenFace, DeepID, ArcFace).

**Note:** DeepFace provides built-in gender classification. Requires face visibility but handles model loading automatically.

```python
from deepface import DeepFace
import cv2

# 1. Initialize DeepFace (models download automatically on first use)
# Backend options: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface'
DETECTOR_BACKEND = 'opencv'

# 2. Gender definitions
# DeepFace uses pre-trained VGG model for gender classification
# No additional training needed - returns 'Man' or 'Woman' with confidence
# Models are automatically downloaded from GitHub releases

# 3. Recognition method
def recognize_person_gender(image_path: str) -> tuple[bool, float, float]:
    """Returns (has_full_human, male_factor, female_factor)"""
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    has_full_human = False
    male_factor, female_factor = 0.0, 0.0

    try:
        # Analyze faces in image
        results = DeepFace.analyze(
            img_path=image_path,
            actions=['gender'],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )

        if results:
            result = results[0] if isinstance(results, list) else results

            # Check if face region suggests full body visible
            region = result.get('region', {})
            face_height = region.get('h', 0)
            face_y = region.get('y', 0)

            # Heuristic: if face is in upper portion, body likely visible below
            if face_y < img_height * 0.4 and face_height < img_height * 0.3:
                has_full_human = True

            # Extract gender probabilities
            gender_data = result.get('gender', {})
            male_prob = gender_data.get('Man', 0) / 100.0
            female_prob = gender_data.get('Woman', 0) / 100.0

            # Scale to 0-2 likeness factor
            male_factor = male_prob * 2.0
            female_factor = female_prob * 2.0

    except Exception:
        pass

    return has_full_human, male_factor, female_factor

# Usage
result = recognize_person_gender("test_image.jpg")
print(f"Human detected: {result[0]}, Male: {result[1]:.2f}, Female: {result[2]:.2f}")
```

---

## 5. PyTorch + torchvision

**Use Cases:** Deep learning framework with extensive computer vision support. Used for custom model training, transfer learning, and deploying pre-trained models for classification, detection, and segmentation.

**Note:** Requires training or loading pre-trained models. Most flexible but requires more setup. Example uses Faster R-CNN for detection and ResNet for gender classification.

```python
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import torch.nn as nn

# 1. Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Person detector (pre-trained on COCO)
person_detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
person_detector.eval().to(device)

# 2. Gender classifier definition
# Requires training on gender-labeled dataset (e.g., CelebA, LFW+attributes)
# Training data needed: ~10,000+ labeled male/female images
# Fine-tune ResNet18 for binary classification
class GenderClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(512, 2)  # 2 classes: male, female

    def forward(self, x):
        return torch.softmax(self.backbone(x), dim=1)

gender_classifier = GenderClassifier().eval().to(device)
# Load trained weights: gender_classifier.load_state_dict(torch.load('gender_model.pth'))

# Transforms
detection_transform = transforms.Compose([transforms.ToTensor()])
classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# COCO class index for person
PERSON_LABEL = 1

# 3. Recognition method
def recognize_person_gender(image_path: str) -> tuple[bool, float, float]:
    """Returns (has_full_human, male_factor, female_factor)"""
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size

    has_full_human = False
    male_factor, female_factor = 0.0, 0.0

    # Detect persons
    img_tensor = detection_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        detections = person_detector(img_tensor)[0]

    for i, label in enumerate(detections['labels']):
        if label == PERSON_LABEL and detections['scores'][i] > 0.7:
            box = detections['boxes'][i].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            box_height = y2 - y1

            # Check for full body (significant vertical coverage)
            if box_height > img_height * 0.5:
                has_full_human = True

                # Crop and classify gender
                person_crop = img.crop((x1, y1, x2, y2))
                crop_tensor = classify_transform(person_crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs = gender_classifier(crop_tensor)[0].cpu().numpy()

                male_factor = float(probs[0]) * 2.0
                female_factor = float(probs[1]) * 2.0
                break

    return has_full_human, male_factor, female_factor

# Usage
result = recognize_person_gender("test_image.jpg")
print(f"Human detected: {result[0]}, Male: {result[1]:.2f}, Female: {result[2]:.2f}")
```

---

## 6. TensorFlow + Keras

**Use Cases:** End-to-end deep learning platform for training and deploying models. Strong ecosystem with TensorFlow Hub for pre-trained models. Used for image classification, object detection, and custom model development.

**Note:** Similar to PyTorch, requires model setup. Example uses TensorFlow Hub models for detection and classification.

```python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# 1. Initialize TensorFlow models
# Person detector from TensorFlow Hub (SSD MobileNet)
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# 2. Gender classifier definition
# Requires training dataset (CelebA, IMDB-Wiki, etc.)
# Training: model.fit(train_images, train_labels, epochs=20)
# Simplified model architecture for gender classification
gender_model = tf.keras.Sequential([
    tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    ),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation='softmax')  # [male, female]
])
# Load weights: gender_model.load_weights('gender_classifier.h5')

# COCO label for person
PERSON_CLASS = 1

# 3. Recognition method
def recognize_person_gender(image_path: str) -> tuple[bool, float, float]:
    """Returns (has_full_human, male_factor, female_factor)"""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]

    has_full_human = False
    male_factor, female_factor = 0.0, 0.0

    # Prepare image for detection
    input_tensor = tf.convert_to_tensor(img_array)[tf.newaxis, ...]

    # Run detection
    results = detector(input_tensor)

    boxes = results['detection_boxes'][0].numpy()
    classes = results['detection_classes'][0].numpy().astype(int)
    scores = results['detection_scores'][0].numpy()

    for i, cls in enumerate(classes):
        if cls == PERSON_CLASS and scores[i] > 0.5:
            y1, x1, y2, x2 = boxes[i]
            box_height = (y2 - y1) * img_height

            if box_height > img_height * 0.5:
                has_full_human = True

                # Crop person region
                y1_px, x1_px = int(y1 * img_height), int(x1 * img_width)
                y2_px, x2_px = int(y2 * img_height), int(x2 * img_width)
                person_crop = img_array[y1_px:y2_px, x1_px:x2_px]

                # Preprocess for gender classification
                crop_resized = tf.image.resize(person_crop, (224, 224))
                crop_normalized = tf.keras.applications.mobilenet_v2.preprocess_input(crop_resized)
                crop_batch = tf.expand_dims(crop_normalized, 0)

                # Classify gender
                probs = gender_model.predict(crop_batch, verbose=0)[0]
                male_factor = float(probs[0]) * 2.0
                female_factor = float(probs[1]) * 2.0
                break

    return has_full_human, male_factor, female_factor

# Usage
result = recognize_person_gender("test_image.jpg")
print(f"Human detected: {result[0]}, Male: {result[1]:.2f}, Female: {result[2]:.2f}")
```

---

## 7. Detectron2 (Facebook AI Research)

**Use Cases:** Facebook's next-generation library for object detection and segmentation. State-of-the-art performance on instance segmentation, panoptic segmentation, and keypoint detection. Best for research and production-grade detection.

**Note:** Powerful but heavier setup. Excellent for person detection with pose keypoints. Gender requires additional classifier.

```python
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Initialize Detectron2 predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
person_predictor = DefaultPredictor(cfg)

# 2. Gender classifier (requires training)
# Training data: labeled male/female dataset
# Can use pose keypoints as features or full image crop
class PoseGenderClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 17 keypoints * 3 values (x, y, confidence)
        self.fc = nn.Sequential(
            nn.Linear(51, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, keypoints):
        return self.fc(keypoints.flatten(1))

gender_classifier = PoseGenderClassifier()
# Load weights: gender_classifier.load_state_dict(torch.load('pose_gender.pth'))

# COCO person class
PERSON_CLASS = 0

# 3. Recognition method
def recognize_person_gender(image_path: str) -> tuple[bool, float, float]:
    """Returns (has_full_human, male_factor, female_factor)"""
    img = cv2.imread(image_path)
    img_height = img.shape[0]

    has_full_human = False
    male_factor, female_factor = 0.0, 0.0

    outputs = person_predictor(img)
    instances = outputs["instances"]

    if len(instances) > 0:
        # Filter for person class
        person_mask = instances.pred_classes == PERSON_CLASS
        if person_mask.any():
            boxes = instances.pred_boxes[person_mask].tensor.cpu().numpy()
            keypoints = instances.pred_keypoints[person_mask].cpu()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                box_height = y2 - y1

                if box_height > img_height * 0.5:
                    has_full_human = True

                    # Use keypoints for gender classification
                    kpts = keypoints[i:i+1]  # Shape: (1, 17, 3)
                    with torch.no_grad():
                        probs = gender_classifier(kpts)[0].numpy()

                    male_factor = float(probs[0]) * 2.0
                    female_factor = float(probs[1]) * 2.0
                    break

    return has_full_human, male_factor, female_factor

# Usage
result = recognize_person_gender("test_image.jpg")
print(f"Human detected: {result[0]}, Male: {result[1]:.2f}, Female: {result[2]:.2f}")
```

---

## Summary Table

| Library | Full-Body Detection | Gender Classification | Training Required | Best For |
|---------|--------------------|-----------------------|-------------------|----------|
| OpenCV | Built-in (Haar/DNN) | Requires model | Model download | General CV, lightweight |
| MediaPipe | Excellent | Not included | No (heuristic only) | Real-time pose |
| YOLO (Ultralytics) | Excellent | Custom training | Yes (for gender) | Fast detection |
| DeepFace | Face-based | Built-in | No | Face analysis |
| PyTorch | Via pre-trained | Custom training | Yes | Research, custom models |
| TensorFlow | Via TF Hub | Custom training | Yes | Production deployment |
| Detectron2 | State-of-the-art | Custom training | Yes | Research, segmentation |

---

## Installation Commands

```bash
# OpenCV
pip install opencv-python opencv-contrib-python

# MediaPipe
pip install mediapipe

# YOLO (Ultralytics)
pip install ultralytics

# DeepFace
pip install deepface

# PyTorch + torchvision
pip install torch torchvision

# TensorFlow
pip install tensorflow tensorflow-hub

# Detectron2 (requires PyTorch first)
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
