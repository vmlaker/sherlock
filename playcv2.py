"""Capture webcam video using OpenCV, and playback."""

import datetime
import sys

import numpy as np
import cv2

import util

DEVICE   = int(sys.argv[1])
WIDTH    = int(sys.argv[2])
HEIGHT   = int(sys.argv[3])
DURATION = float(sys.argv[4])

# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

# Create the OpenCV video capture object.
cap = cv2.VideoCapture(DEVICE)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Create the display window.
cv2.namedWindow('hello', cv2.cv.CV_WINDOW_NORMAL)

end = datetime.datetime.now() + datetime.timedelta(seconds=DURATION)
while end > datetime.datetime.now():

    # Take a snapshot show it.
    hello, image = cap.read()        
    cv2.imshow('hello', image)
    cv2.waitKey(1)
    
    # Print the framerate.
    print('%05.3f, %05.3f, %05.3f'%framerate.tick())

# The end.
