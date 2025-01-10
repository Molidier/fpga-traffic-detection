import cv2

#открывает окно где фото конвертировалось в серую верисю
img = cv2.imread('data/traffics.png') #take an image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert the image into gray and save as a new image
cv2.imshow('Grayscale', gray) #show the new created img with the name 'Grayscale'
cv2.waitKey(0) #after pressing key 0
cv2.destroyAllWindows() #close the opened window with the gray image