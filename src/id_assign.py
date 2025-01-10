import cv2
import numpy as np

# Initialize tracker dictionary
tracker = {}  # Format: {id: (x, y, w, h)}
next_id = 1  # To assign unique IDs

# Function to calculate overlap
def calculate_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    center1 = (x1 + w1 // 2, y1 + h1 // 2)
    center2 = (x2 + w2 // 2, y2 + h2 // 2)
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    return distance

# Main video processing loop
cap = cv2.VideoCapture('data/sample_video.mp4')
car_cascade = cv2.CascadeClassifier('data/haarcascade_car.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.175, minNeighbors=3, minSize=(10, 10), maxSize=(300, 300))

    updated_tracker = {}
    for (x, y, w, h) in cars:
        assigned = False
        for id, tracked_box in tracker.items():
            if calculate_overlap((x, y, w, h), tracked_box) < 50:  # Threshold for matching
                updated_tracker[id] = (x, y, w, h)
                assigned = True
                break
        if not assigned:
            updated_tracker[next_id] = (x, y, w, h)
            next_id += 1

    tracker = updated_tracker

    # Draw IDs on the frame
    for id, (x, y, w, h) in tracker.items():
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Vehicle Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
