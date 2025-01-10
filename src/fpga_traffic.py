import serial
import time
import cv2
import numpy as np
from sort import Sort

# Configure the serial port (update the port and baudrate as per your FPGA settings)
ser = serial.Serial(
    port='COM7',  # For Windows: 'COMx', for Linux: '/dev/ttyUSBx'
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=2
)

def send_data(data):
    """Send string data to the FPGA over UART."""
    if ser.is_open:
        ser.write(chr(data).encode())  # Sending the data to FPGA
        print(f"Sent: {data}")


def calculate_histogram(frame, box):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

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



tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)  # Adjust iou_threshold

line_y = 400
vehicle_count = 0
counted_ids = set()
appearance_histograms = {}

#cap = cv2.VideoCapture('data/sample_video.mp4')
#cap = cv2.VideoCapture('data/Cars-Busy-Streets.mp4')
cap = cv2.VideoCapture('data/heavy-traffic.mp4')

#car_cascade = cv2.CascadeClassifier('data/haarcascade_car.xml')
car_cascade = cv2.CascadeClassifier('data/cars.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

    #prepare detections
    detections = [[x, y, x+w, y+h, 1] for (x,y,w,h) in cars]
    tracked_obj = tracker.update(np.array(detections))

    #Count vechiles
    for obj in tracked_obj:
        x1, y1, x2, y2, track_id = map(int, obj)

        #smoothing coordinates
        smoothed_coords = smooth_positions(track_id, [x1, y1, x2, y2])
        x1, y1, x2, y2 = map(int, smoothed_coords)

        if track_id not in appearance_histograms:
            appearance_histograms[track_id] = calculate_histogram(frame, (x1, y1, x2, y2))

        if track_id not in counted_ids and y1 <= line_y <= y2:
            counted_ids.add(track_id)
            vehicle_count +=1

        # Draw box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw line and count
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)
    cv2.putText(frame, f"Count: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #Sending Vehicle count to fpga
    send_data(vehicle_count)


    # Show frame
    cv2.imshow('Vehicle Tracking and Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()