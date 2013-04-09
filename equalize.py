import cv2
cap = cv2.VideoCapture(0)
while True:
    f, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    cv2.imshow('input', image)
    cv2.waitKey(1)
