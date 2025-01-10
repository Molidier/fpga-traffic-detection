import cv2

#load video

video_path = 'data/sample_video.mp4'
cap = cv2.VideoCapture(video_path) #create tuple

frame_number = 0

while cap.isOpened():
    isTrue, frame = cap.read()
    if not isTrue:
        break
    cv2.imshow('traffic_video', frame)

    #resize frames to reduce processing time
    '''
    Examples:
    frame_1080p = cv2.resize(frame, (1920, 1080))  # Full HD
    frame_720p = cv2.resize(frame, (1280, 720))    # HD
    frame_480p = cv2.resize(frame, (640, 480))     # Standard Definition
    frame_360p = cv2.resize(frame, (640, 360))     # Widescreen SD
    frame_240p = cv2.resize(frame, (320, 240))     # QVGA
    '''
    frame = cv2.resize(frame, (640, 360)) #качество видео -> fewer pixels

    #add grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray_traffic', gray)

    #apply gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    #Thresholding -> convert grayscale into binary img
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #127 - threshold val -> pixels with intensity > 255(white), others 0 (black)
    #255 ->. max pixel value to assign
    #cv2.THRESH_BINARY: The thresholding type that converts the image to black and white.


    #detect edges using Canny Edge Detection
    edges = cv2.Canny(gray, 50, 150)
    #150 -> upper threshold -> if p > max -> edge
    #50 -> if p < min -> ignored

    #detect shapes in the frames
    #boundaries of a detected shapes
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #return sequence list and MatLike

    #countor -> NumPy array
    #Draw contours
    for contour in contours:
        if cv2.contourArea(contour) > 300 and cv2.contourArea(contour) < 400: #Filter small contours
        #smaller -> most likely noise
            x, y, w, h = cv2.boundingRect(contour)
            # x, y -> coordinates of the top-left corner of the rectangle

            #Aspect ratio filtering
            aspect_ratio = float(w)/h
            if 0.5 < aspect_ratio < 2.0:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
                #(x+w, y+h) -> bottom right corner of the rectangle
                #(0, 255, 0) -> the color of the rectangle in BGR format (green in this case)
                #2 -> thickness of the rectangle's border

                # Add text annotation
                cv2.putText(frame, f"AR: {aspect_ratio:.2f}", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            
    #Show the result
    cv2.imshow('Contours', frame)

    #Save every 10th frame as an image
    if frame_number % 5 == 0:
        cv2.imwrite(f'output/traffic_frame_{frame_number}.jpg', frame)
    frame_number += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()