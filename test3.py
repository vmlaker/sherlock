"""Multiple process image processing using mid-stage result."""

import datetime
import sys
import cv2
import numpy as np

import mpipe
import util
import iproc

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

# Keep track of previous iteration's timestamp.
tstamp_prev = None  

def step1(image):
    """Return preprocessed image."""
    global tstamp_prev
    alpha, tstamp_prev = iproc.getAlpha(tstamp_prev)
    image_pre = iproc.preprocess2(image)
    return (image, image_pre, alpha)
 
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

        self.putResult((image, image_diff))

        # Accumulate.
        hello = cv2.accumulateWeighted(
            image_gray,
            image_acc,
            alpha,
            )
        
def step3((image, image_diff)):
    """Postprocess image using given difference."""
    iproc.postprocess(image, image_diff)
    return image

def showImage(image):
    """Display the result of processing."""
    # Show the images.
    for name in NAMES:
        exec('the_image = %s'%name)
        cv2.imshow(name, the_image)
    cv2.waitKey(1)
    return True

# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

def printStatus(hello):
    # Print the framerate.
    global framerate
    print('%05.3f, %05.3f, %05.3f'%framerate.tick())

stages = list()
stages.append(mpipe.OrderedStage(step1))
stages.append(mpipe.Stage(Step2Worker))
stages.append(mpipe.OrderedStage(step3))
stages.append(mpipe.OrderedStage(showImage))
stages.append(mpipe.OrderedStage(printStatus))
stages[0].link(stages[1])
stages[1].link(stages[2])
stages[2].link(stages[3])
stages[3].link(stages[4])
pipe = mpipe.Pipeline(stages[0])

end = datetime.datetime.now() + datetime.timedelta(seconds=DURATION)
while end > datetime.datetime.now():
    hello, image = cap.read()
    pipe.put(image)
pipe.put(None)

# The end.
