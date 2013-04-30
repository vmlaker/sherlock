"""Multiprocess motion detection."""

import datetime
import sys
import cv2
import numpy as np

import mpipe
import util
import iproc

DEVICE   = int(sys.argv[1])
WIDTH    = int(sys.argv[2])
HEIGHT   = int(sys.argv[3])
DURATION = float(sys.argv[4])  # In seconds.

# Create the OpenCV video capture object.
cap = cv2.VideoCapture(DEVICE)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Create the output window.
cv2.namedWindow('motion 2', cv2.cv.CV_WINDOW_NORMAL)

# Maintain accumulation of thresholded differences.
image_acc = None  

# Keep track of previous iteration's timestamp.
tstamp_prev = None  

def step1(image):
    """Return preprocessed image."""
    global tstamp_prev
    alpha, tstamp_prev = iproc.getAlpha(tstamp_prev)
    image_pre = iproc.preprocess(image)
    return (image, image_pre, alpha)
 
def step2((image, image_gray, alpha)):
    """Compute difference between given image and accumulation,
    then accumulate and return the difference. Initialize accumulation
    if needed (if opacity is 100%.)"""

    # Initalize accumulation if so indicated.
    global image_acc
    if image_acc is None:
        image_acc = np.empty(np.shape(image_gray))

    # Compute difference.
    image_diff = cv2.absdiff(
        image_acc.astype(image_gray.dtype),
        image_gray,
        )

    # Accumulate.
    hello = cv2.accumulateWeighted(
        image_gray,
        image_acc,
        alpha,
        )

    return (image, image_diff)

# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

def step3((image, image_diff)):
    """Postprocess image using given difference."""
    iproc.postprocess(image, image_diff)

    # Write the framerate on top of the image.
    iproc.writeOSD(
        image, 
        ('%.2f, %.2f, %.2f fps'%framerate.tick(),),
        ratio=0.04,
        )
    return image

def step4(image):
    """Display the image."""
    cv2.imshow('motion 2', image)
    cv2.waitKey(1)  # Allow HighGUI to process event.

stage1 = mpipe.OrderedStage(step1)
stage2 = mpipe.OrderedStage(step2)
stage3 = mpipe.OrderedStage(step3)
stage4 = mpipe.OrderedStage(step4)
stage1.link(stage2)
stage2.link(stage3)
stage3.link(stage4)
pipe = mpipe.Pipeline(stage1)

end = datetime.datetime.now() + datetime.timedelta(seconds=DURATION)
while end > datetime.datetime.now():
    hello, image = cap.read()
    pipe.put(image)
pipe.put(None)

# The end.
