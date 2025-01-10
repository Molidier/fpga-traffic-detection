import cv2
import numpy as np

import sys
sys.path.append(r'C:\Users\user\Desktop\ML\TrafficDetection\deep_sort')

from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from application_util.preprocessing import non_max_suppression
from tools.generate_detections import create_box_encoder
from scipy.optimize import linear_sum_assignment  # Updated import

# Function to calculate histogram for appearance features
def calculate_histogram(frame, box):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

# Smoothing parameters
alpha = 0.8  # Smoothing factor (0.0 to 1.0, higher is smoother)
smoothed_boxes = {}

def smooth_positions(track_id, current_box):
    global smoothed_boxes
    if track_id not in smoothed_boxes:
        smoothed_boxes[track_id] = current_box
    else:
        smoothed_boxes[track_id] = [
            alpha * current + (1 - alpha) * previous
            for current, previous in zip(current_box, smoothed_boxes[track_id])
        ]
    return smoothed_boxes[track_id]

# Initialize Deep SORT
max_cosine_distance = 0.3  # Adjust based on performance
nn_budget = 100
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Load feature encoder for ReID
encoder_model_path = './resources/networks/mars-small128.pb'
encoder = create_box_encoder(encoder_model_path, batch_size=32)

# Line position and counters
line_y = 400
vehicle_count = 0
counted_ids = set()
appearance_histograms = {}

# Open video
cap = cv2.VideoCapture('data/sample_video.mp4')
car_cascade = cv2.CascadeClassifier('data/cars.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect cars using Haar cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))

    # Prepare detections for Deep SORT
    bboxes = [[x, y, x + w, y + h] for (x, y, w, h) in cars]
    confidences = [1.0] * len(bboxes)  # Haar cascade doesn't provide confidence, so use constant
    features = encoder(frame, bboxes)
    detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(bboxes, confidences, features)]

    # Non-max suppression to remove overlapping detections
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = non_max_suppression(boxes, 1.0, scores)
    detections = [detections[i] for i in indices]

    # Update tracker with current detections
    tracker.predict()
    tracker.update(detections)

    # Count vehicles and draw tracks
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        # Get bounding box and track ID
        bbox = track.to_tlbr()
        track_id = track.track_id

        # Smooth bounding box coordinates
        smoothed_coords = smooth_positions(track_id, bbox)
        x1, y1, x2, y2 = map(int, smoothed_coords)

        # Compute and store appearance histogram
        if track_id not in appearance_histograms:
            appearance_histograms[track_id] = calculate_histogram(frame, (x1, y1, x2, y2))

        # Count vehicles crossing the line
        if track_id not in counted_ids and y1 <= line_y <= y2:
            counted_ids.add(track_id)
            vehicle_count += 1

        # Draw bounding box and track ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw line and display count
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)
    cv2.putText(frame, f"Count: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow('Vehicle Tracking and Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
