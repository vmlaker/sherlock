"""Single process image processing."""

import datetime
import sys
import cv2
import numpy as np

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

# Create the output windows.
NAMES = (
    'image',
#    'image_pre', 
#    'image_diff', 
#    'image_thresh',
#    'image_acc',
    )
for name in NAMES:
    cv2.namedWindow(name, cv2.cv.CV_WINDOW_NORMAL)

# Accumulation of thresholded differences.
image_acc = None  

def step1(image):
    """Return preprocessed image."""
    return iproc.preprocess2(image)
 
def step2(image, alpha):
    """Compute difference between given image and accumulation,
    then accumulate and return the difference. Initialize accumulation
    if needed (if opacity is 100%.)"""

    # Use global accumulation.
    global image_acc

    # Initalize accumulation if so indicated.
    if alpha == 1.0:
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

    return image_diff

def step3(image, image_diff):
    """Postprocess image using given difference."""
    iproc.postprocess(image, image_diff)

# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

def step4():
    """Display the result of processing."""
    # Show the images.
    for name in NAMES:
        exec('the_image = %s'%name)
        cv2.imshow(name, the_image)
    cv2.waitKey(1)

    # Print the framerate.
    global framerate
    print('%05.3f, %05.3f, %05.3f'%framerate.tick())

# Keep track of previous iteration's timestamp.
tstamp_prev = None  

end = datetime.datetime.now() + datetime.timedelta(seconds=DURATION)
while end > datetime.datetime.now():

    # Take a snapshot and mark the snapshot time.
    hello, image = cap.read()

    # Compute alpha value.
    alpha, tstamp_prev = iproc.getAlpha(tstamp_prev)

    # Preprocess the image.
    image_pre = step1(image)
    
    # Compute difference, and accumulate.
    image_diff = step2(image_pre, alpha)

    # Augment.
    step3(image, image_diff)

    # Display.
    step4()

# The end.
