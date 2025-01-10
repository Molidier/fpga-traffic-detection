import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist
from PIL import Image
import torchreid

# Check for GPU and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the ReID model
model = torchreid.models.build_model(
    name='osnet_x0_25',
    num_classes=1000,
    pretrained=True
)
model.eval().to(device)

# Preprocess input for the ReID model
preprocess = torchreid.data.transforms.build_transforms(
    height=256, width=128, 
    norm_mean=[0.485, 0.456, 0.406], 
    norm_std=[0.229, 0.224, 0.225]
)[0]

# Initialize track memory for ReID
track_memory = {}  # Format: {id: (features, bbox)}

# Function to extract features for a batch of cropped images
def extract_features_batch(images):
    tensors = torch.stack([preprocess(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))) for img in images]).to(device)
    with torch.no_grad():
        features = model(tensors).cpu().numpy()
    return features

# Function to match features and assign IDs
def match_features(new_features, new_boxes, track_memory, threshold=0.5):
    old_features = np.vstack([features for features, _ in track_memory.values()]) if track_memory else np.empty((0, new_features.shape[1]))
    assigned_ids = []
    used_ids = set()

    if old_features.size > 0:
        distances = cdist(new_features, old_features, metric='cosine')
        for i, row in enumerate(distances):
            min_dist = row.min()
            best_id = np.argmin(row)
            if min_dist < threshold and best_id not in used_ids:
                assigned_ids.append((new_boxes[i], best_id))
                track_memory[best_id] = (new_features[i], new_boxes[i])
                used_ids.add(best_id)
            else:
                new_id = max(track_memory.keys(), default=0) + 1
                track_memory[new_id] = (new_features[i], new_boxes[i])
                assigned_ids.append((new_boxes[i], new_id))
                used_ids.add(new_id)
    else:
        for i, features in enumerate(new_features):
            new_id = max(track_memory.keys(), default=0) + 1
            track_memory[new_id] = (features, new_boxes[i])
            assigned_ids.append((new_boxes[i], new_id))

    return assigned_ids

# Load video and Haar Cascade
cap = cv2.VideoCapture('data/sample_video.mp4')
car_cascade = cv2.CascadeClassifier('data/haarcascade_car.xml')

frame_count = 0
frame_skip = 2  # Process every 3rd frame for efficiency

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for performance
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.175, minNeighbors=3, minSize=(10, 10), maxSize=(300, 300))

    # Prepare data for ReID feature extraction
    new_features = []
    new_boxes = []
    cropped_images = []
    for (x, y, w, h) in cars:
        cropped = frame[y:y + h, x:x + w]
        if cropped.shape[0] > 0 and cropped.shape[1] > 0:
            cropped_images.append(cropped)
            new_boxes.append((x, y, w, h))

    if cropped_images:
        new_features = extract_features_batch(cropped_images)
        assigned_ids = match_features(new_features, new_boxes, track_memory)

        # Draw tracked objects
        for (x, y, w, h), track_id in assigned_ids:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display frame
    cv2.imshow('Vehicle Tracking with ReID', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
