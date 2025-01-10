import cv2
import numpy as np

def change_brightness(frame, factor):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], factor)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_motion_blur(frame, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(frame, -1, kernel)

cap = cv2.VideoCapture('data/sample_video.mp4')
#cap = cv2.VideoCapture('data/Cars-Busy-Streets.mp4')
#cap = cv2.VideoCapture('data/heavy-traffic.mp4')

#car_cascade = cv2.CascadeClassifier('data/haarcascade_car.xml')
car_cascade = cv2.CascadeClassifier('data/cars.xml')

frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    #frame = change_brightness(frame, 50)
    #frame = apply_motion_blur(frame, 15)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #???
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30), maxSize=(300, 300))
    #scale factor -> smaller -> more precise -> computationally expensive
    #lower values -> increase sensitivity -> more false positives

    #show the # of detected vehicles

    for (x, y, w, h) in cars:
        aspect_ratio = w / h
        if 0.7 < aspect_ratio < 4.0:  # Typical aspect ratio for vehicles
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            #count cars
            vechile_count = len(cars)
            cv2.putText(
                frame,
                f'Vehicles: {vechile_count}',
                (10,50), #pos of text on the frame
                cv2.FONT_HERSHEY_SIMPLEX, #font type
                1, #size of the text
                (0,255,0),
                2 #thickness of the text
            )

    #cv2.rectangle(frame, (100, 300), (500, 400), (255, 0, 0), 2)
    cv2.imshow('Vehicle Detection', frame)

        #Save every 10th frame as an image
    # if frame_number % 5 == 0:
    #     cv2.imwrite(f'output/traffic_frame_1.25_3_{frame_number}.jpg', frame)
    # frame_number += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()
