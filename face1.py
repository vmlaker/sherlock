"""Sequential, vanilla face detection."""

import datetime
import sys
import cv2
import numpy as np

import util
import cascade

DEVICE   = int(sys.argv[1])
WIDTH    = int(sys.argv[2])
HEIGHT   = int(sys.argv[3])
DURATION = float(sys.argv[4])  # In seconds.

# Create the OpenCV video capture object.
cap = cv2.VideoCapture(DEVICE)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Monitor framerates.
framerate = util.RateTicker((1,5,10))

# Allow view window to be resizeable.
cv2.namedWindow('face detection', cv2.cv.CV_WINDOW_NORMAL)

end = datetime.datetime.now() + datetime.timedelta(seconds=DURATION)
while end > datetime.datetime.now():

    hello, image = cap.read()

    size = np.shape(image)[:2]
    result = list()
    for classi in cascade.classifiers:
        rects = classi.detectMultiScale(
            image,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=tuple([x/20 for x in size]),
            maxSize=tuple([x/2 for x in size]),
            )
        if len(rects):
            for a,b,c,d in rects:
                result.append((a,b,c,d, cascade.colors[classi]))
    for x1, y1, x2, y2, color in result:
        cv2.rectangle(
            image,
            (x1, y1), (x1+x2, y1+y2), 
            color=color,
            thickness=2,
            )
    scale = 0.85
    for org, text in (
        ((20, int(30*scale)), '%dx%d'%(size[1], size[0])),
        ((20, int(60*scale)), '%.2f, %.2f, %.2f'%framerate.tick()),
        ):
        cv2.putText(
            image,
            text=text,
            org=org,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale,
            color=(0,255,0),
            thickness=2,
            )
    cv2.imshow('face detection', image)
    cv2.waitKey(1)

# The end.
