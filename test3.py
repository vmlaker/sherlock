"""Multiple process image processing using mid-stage result."""

import datetime
import sys
import cv2
import numpy as np

import mpipe
import util

WIDTH  = int(sys.argv[1])
HEIGHT = int(sys.argv[2])
DURATION = float(sys.argv[3])  # In seconds.

# Create the OpenCV video capture object.
cap = cv2.VideoCapture(0)
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

tstamp_prev = None  # Keep track of previous iteration's timestamp.
def getAlpha():
    """Return alpha value based on elapsed time."""
    global tstamp_prev
    now = datetime.datetime.now()
    alpha = 1.0
    if tstamp_prev:
        tdelta = now - tstamp_prev
        alpha = tdelta.total_seconds()
        alpha *= 0.50  # Halve the alpha value -- looks better.
    tstamp_prev = now
    return alpha        

def step1(image):
    """Return preprocessed image."""
    alpha = getAlpha()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (image, cv2.equalizeHist(image_gray), alpha)
 
class Step2Worker(mpipe.OrderedWorker):
    def doTask(self, (image, image_gray, alpha)):
        """Compute difference between given image and accumulation,
        then accumulate and set result with the difference. 
        Initialize accumulation if needed (if opacity is 100%.)"""
        
        # Use global accumulation.
        global image_acc

        # Initalize accumulation if so indicated.
        if alpha == 1.0:
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
        
        self.putResult((image, image_diff))

def step3((image, image_diff)):
    """Augment given image with given difference."""

    # Apply threshold.
    hello, image_thresh = cv2.threshold(
        image_diff,
        thresh=35,
        maxval=255,
        type=cv2.THRESH_BINARY,
        )

    # Find contours.
    contours, hier = cv2.findContours(
        image_thresh,
        #np.copy(image_thresh),
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE,
        )        

    # Sort and filter contours.
    # Use a sensible threshold value based on image resolution.
    area_threshold = image_thresh.shape[0] * image_thresh.shape[1]
    area_threshold *= 0.00005 /2
    contours = sorted(
        contours, 
        key=lambda x: cv2.contourArea(x), 
        reverse=True)
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Since contours are sorted, we can safely break out 
        # of the loop once area falls below threshold.
        if area < area_threshold:
            break

        # Add this contour to the collection.
        filtered.append(contour)

    # Augment output image with contours.
    cv2.drawContours(
        image,
        filtered,
        -1,
        color=(0, 254, 254),  # Yellow.
        thickness=2,
        )

    # Augment output image with rectangles.
    for contour in filtered:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(
            image,
            (x,y),
            (x+w,y+h),
            color=(0, 254, 0),
            thickness=2,
            )

    return image

# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

def step4(image):

    # Display the images.
    for name in NAMES:
        exec('the_image = %s'%name)
        cv2.imshow(name, the_image)
    cv2.waitKey(1)

    # Print the framerate.
    global framerate
    print('%05.3f, %05.3f, %05.3f'%framerate.tick())

stage1 = mpipe.OrderedStage(step1)
stage2 = mpipe.Stage(Step2Worker)
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
