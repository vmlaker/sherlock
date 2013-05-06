"""Difference from running average
with multiprocessing and shared memory
utilizing mid-stage result."""

import multiprocessing
import datetime
import time
import sys
import cv2
import numpy as np
import mpipe
import sharedmem

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
cv2.namedWindow('diff average 4', cv2.cv.CV_WINDOW_NORMAL)

# Create a process-shared (common) table keyed on timestamps
# and holding references to allocated memory and other useful values.
manager = multiprocessing.Manager()
common = manager.dict()

# Maintain accumulation of thresholded differences.
image_acc = None  

# Keep track of previous iteration's timestamp.
tstamp_prev = None  

class Step1(mpipe.OrderedWorker):
    def doTask(self, tstamp):
        """Compute difference between given image and accumulation,
        then accumulate and set result with the difference. 
        Initialize accumulation if needed (if opacity is 100%.)"""

        # Compute the alpha value.
        global tstamp_prev
        alpha, tstamp_prev = util.getAlpha(tstamp_prev)

        image = common[tstamp]['image_in']
        
        # Initalize accumulation if so indicated.
        global image_acc
        if image_acc is None:
            image_acc = np.empty(np.shape(image))

        # Allocate shared memory for the diff image.
        shape = np.shape(image)
        dtype = image.dtype
        image_diff = sharedmem.empty(shape, dtype)
        
        # Compute difference.
        cv2.absdiff(
            image_acc.astype(image.dtype),
            image,
            image_diff,
            )

        hello = common[tstamp]
        hello['image_diff'] = image_diff
        common[tstamp] = hello
        
        self.putResult(tstamp)

        # Accumulate.
        hello = cv2.accumulateWeighted(
            image,
            image_acc,
            alpha,
            )

# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

def step2(tstamp):
    """Postprocess image using given difference."""

    # Write the framerate on top of the image.
    util.writeOSD(
        common[tstamp]['image_diff'],
        ('%.2f, %.2f, %.2f fps'%framerate.tick(),),
        )
    cv2.imshow('diff average 4', common[tstamp]['image_diff'])
    cv2.waitKey(1)  # Allow HighGUI to process event.
    return tstamp

def step3(tstamp):
    """Make sure the timestamp is at least a certain 
    age before propagating it further."""
    delta = datetime.datetime.now() - tstamp
    duration = datetime.timedelta(seconds=2) - delta
    if duration > datetime.timedelta():
        time.sleep(duration.total_seconds())
    return tstamp

stage1 = mpipe.Stage(Step1)
stage2 = mpipe.OrderedStage(step2)
stage3 = mpipe.OrderedStage(step3)
stage1.link(stage2)
stage2.link(stage3)
pipe = mpipe.Pipeline(stage1)

# Create an auxiliary process (modeled as a one-task pipeline)
# that simply pulls results from the image processing pipeline, 
# and deallocates the associated shared memory.
def deallocate(task):
    for tstamp in pipe.results():
        del common[tstamp]
pipe2 = mpipe.Pipeline(mpipe.UnorderedStage(deallocate))
pipe2.put(True)  # Start it up right away.

now = datetime.datetime.now()
end = now + datetime.timedelta(seconds=DURATION)
while end > now:
    now = datetime.datetime.now()
    hello, image = cap.read()

    # Allocate shared memory for a copy of the input image.
    shape = np.shape(image)
    dtype = image.dtype
    image_in   = sharedmem.empty(shape, dtype)
    
    # Copy the input image to it's shared memory version.
    image_in[:] = image.copy()
    
    common[now] = {
        'image_in'   : image_in,
        }
    pipe.put(now)

# Signal processing pipeline to stop.
pipe.put(None)

# Signal deallocator to stop and wait until it frees all memory.
pipe2.put(None)
for result in pipe2.results():
    pass

# The end.
