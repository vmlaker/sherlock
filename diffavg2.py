"""Difference from running average
with multiprocessing."""

import datetime
import sys
import cv2
import numpy as np
import mpipe

import coils
import util

DEVICE   = int(sys.argv[1])
WIDTH    = int(sys.argv[2])
HEIGHT   = int(sys.argv[3])
DURATION = float(sys.argv[4])  # In seconds.

class Step1(mpipe.OrderedWorker):
    def __init__(self):
        self.image_acc = None  # Maintain accumulation of thresholded differences.
        self.tstamp_prev = None  # Keep track of previous iteration's timestamp.

    def doTask(self, image):
        """Compute difference between given image and accumulation,
        then accumulate and return the difference. Initialize accumulation
        if needed (if opacity is 100%.)"""
        
        # Compute the alpha value.
        alpha, self.tstamp_prev = util.getAlpha(self.tstamp_prev)

        # Initalize accumulation if so indicated.
        if self.image_acc is None:
            self.image_acc = np.empty(np.shape(image))

        # Compute difference.
        image_diff = cv2.absdiff(
            self.image_acc.astype(image.dtype),
            image,
            )

        # Accumulate.
        hello = cv2.accumulateWeighted(
            image,
            self.image_acc,
            alpha,
            )

        return image_diff

# Monitor framerates for the given seconds past.
framerate = coils.RateTicker((1,5,10))

# Create the output window.
cv2.namedWindow('diff average 2', cv2.cv.CV_WINDOW_NORMAL)

def step2(image):
    """Display the image, stamped with framerate."""
    util.writeOSD(image, ('%.2f, %.2f, %.2f fps'%framerate.tick(),),)
    cv2.imshow('diff average 2', image)
    cv2.waitKey(1)  # Allow HighGUI to process event.

# Assemble the pipeline.
stage1 = mpipe.Stage(Step1)
stage2 = mpipe.OrderedStage(step2)
stage1.link(stage2)
pipe = mpipe.Pipeline(stage1)

# Create the OpenCV video capture object.
cap = cv2.VideoCapture(DEVICE)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Run the video capture loop, feeding the image processing pipeline.
end = datetime.datetime.now() + datetime.timedelta(seconds=DURATION)
while end > datetime.datetime.now():
    hello, image = cap.read()
    pipe.put(image)

# Signal processing pipeline to stop.
pipe.put(None)

# The end.
