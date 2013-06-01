"""Difference from running average."""

import datetime
import sys
import cv2
import numpy as np

import coils
import util

DEVICE   = int(sys.argv[1])
WIDTH    = int(sys.argv[2])
HEIGHT   = int(sys.argv[3])
DURATION = float(sys.argv[4])  # In seconds.

# Create the OpenCV video capture object.
cap = cv2.VideoCapture(DEVICE)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Create the output window.
cv2.namedWindow('diff average 1', cv2.cv.CV_WINDOW_NORMAL)

# Maintain accumulation of thresholded differences.
image_acc = None  

# Keep track of previous iteration's timestamp.
tstamp_prev = None  

# Monitor framerates for the given seconds past.
framerate = coils.RateTicker((1,5,10))

# Run the loop for designated amount of time.
end = datetime.datetime.now() + datetime.timedelta(seconds=DURATION)
while end > datetime.datetime.now():

    # Take a snapshot and mark the snapshot time.
    hello, image = cap.read()

    # Compute alpha value.
    alpha, tstamp_prev = util.getAlpha(tstamp_prev)

    # Initalize accumulation if so indicated.
    if image_acc is None:
        image_acc = np.empty(np.shape(image))

    # Compute difference.
    image_diff = cv2.absdiff(
        image_acc.astype(image.dtype),
        image,
        )

    # Accumulate.
    hello = cv2.accumulateWeighted(
        image,
        image_acc,
        alpha,
        )

    # Write the framerate on top of the image.
    util.writeOSD(image_diff, ('%.2f, %.2f, %.2f fps'%framerate.tick(),),)

    # Display the image.
    cv2.imshow('diff average 1', image_diff)

    # Allow HighGUI to process event.
    cv2.waitKey(1)

# The end.
