"""Sequential, vanilla object detection."""

import datetime
import sys
import cv2
import numpy as np

import util
import iproc
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
cv2.namedWindow('object detection', cv2.cv.CV_WINDOW_NORMAL)

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

    # Draw the rectangles.
    for x1, y1, x2, y2, color in result:
        cv2.rectangle(
            image,
            (x1, y1), (x1+x2, y1+y2), 
            color=color,
            thickness=2,
            )

    # Write image dimensions and framerate.
    iproc.writeOSD(
        image,
        ('%dx%d'%(size[1], size[0]),
         '%.2f, %.2f, %.2f fps'%framerate.tick()),
        ratio=0.04,
        )

    cv2.imshow('object detection', image)
    cv2.waitKey(1)

# The end.
