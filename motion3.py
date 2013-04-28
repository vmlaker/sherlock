"""Multiple process image processing using mid-stage result."""

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
cv2.namedWindow('motion 3', cv2.cv.CV_WINDOW_NORMAL)

# Maintain accumulation of thresholded differences.
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
        
        # Initalize accumulation if so indicated.
        global image_acc
        if image_acc is None:
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

def showImage(image):
    """Display the image."""
    cv2.imshow('motion 3', image)
    cv2.waitKey(1)  # Allow HighGUI to process event.

stages = list()
stages.append(mpipe.OrderedStage(step1))
stages.append(mpipe.Stage(Step2Worker))
stages.append(mpipe.OrderedStage(step3))
stages.append(mpipe.OrderedStage(showImage))
stages[0].link(stages[1])
stages[1].link(stages[2])
stages[2].link(stages[3])
pipe = mpipe.Pipeline(stages[0])

end = datetime.datetime.now() + datetime.timedelta(seconds=DURATION)
while end > datetime.datetime.now():
    hello, image = cap.read()
    pipe.put(image)
pipe.put(None)

# The end.
