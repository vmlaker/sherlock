import cv2
cap = cv2.VideoCapture(0)
while True:
    f, image = cap.read()
    cv2.imshow('input', image)
    cv2.waitKey(1)
