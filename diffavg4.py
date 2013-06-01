"""Difference from running average
with multiprocessing and shared memory
using filter to resolve bottleneck."""

import multiprocessing
import datetime
import time
import sys
import cv2
import numpy as np
import mpipe
import sharedmem

import coils
import util

DEVICE   = int(sys.argv[1])
WIDTH    = int(sys.argv[2])
HEIGHT   = int(sys.argv[3])
DURATION = float(sys.argv[4])  # In seconds.

# Create a process-shared (common) table keyed on timestamps
# and holding references to allocated memory.
manager = multiprocessing.Manager()
common = manager.dict()

class Step1(mpipe.OrderedWorker):
    def __init__(self):
        self.image_acc = None  # Maintain accumulation of thresholded differences.
        self.tstamp_prev = None  # Keep track of previous iteration's timestamp.
        self.framerate = coils.RateTicker((1,5,10))  # Monitor framerates.

    def doTask(self, tstamp):
        """Compute difference between given image and accumulation,
        then accumulate and set result with the difference. 
        Initialize accumulation if needed (if opacity is 100%.)"""

        # Compute the alpha value.
        alpha, self.tstamp_prev = util.getAlpha(self.tstamp_prev)

        image = common[tstamp]['image_in']
        
        # Initalize accumulation if so indicated.
        if self.image_acc is None:
            self.image_acc = np.empty(np.shape(image))

        # Allocate shared memory for the diff image.
        shape = np.shape(image)
        dtype = image.dtype
        image_diff = sharedmem.empty(shape, dtype)
        
        # Compute difference.
        cv2.absdiff(
            self.image_acc.astype(image.dtype),
            image,
            image_diff,
            )

        util.writeOSD(
            image_diff,
            ('',  # Keep first line empty (will be written to later.)
             '(%.2f, %.2f, %.2f fps process)'%self.framerate.tick(),),
        )
        
        # Write diff image (actually, reference thereof) to process-shared table.
        hello = common[tstamp]
        hello['image_diff'] = image_diff
        common[tstamp] = hello
        
        # Propagate result to the next stage.
        self.putResult(tstamp)

        # Accumulate.
        hello = cv2.accumulateWeighted(
            image,
            self.image_acc,
            alpha,
            )

# Monitor framerates for the given seconds past.
framerate2 = coils.RateTicker((1,5,10))

# Create the output window.
cv2.namedWindow('diff average 4', cv2.cv.CV_WINDOW_NORMAL)

def step2(tstamp):
    """Display the image, stamped with framerate."""
    util.writeOSD(
        common[tstamp]['image_diff'],
        ('%.2f, %.2f, %.2f fps'%framerate2.tick(),),
        )
    cv2.imshow('diff average 4', common[tstamp]['image_diff'])
    cv2.waitKey(1)  # Allow HighGUI to process event.
    return tstamp

# Assemble the pipeline.
stage1 = mpipe.Stage(Step1)
stage2 = mpipe.FilterStage(
    (mpipe.OrderedStage(step2),),
    max_tasks=2,  # Allow maximum 2 tasks in the viewer stage.
    drop_results=True,
    )
stage1.link(stage2)
pipe = mpipe.Pipeline(
    mpipe.FilterStage(
        (stage1,),
        max_tasks=3,  # Allow maximum 3 tasks in pipeline.
        drop_results=True,
        )
    )

# Create an auxiliary process (modeled as a one-task pipeline)
# that simply pulls results from the image processing pipeline, 
# and deallocates associated shared memory after allowing
# the designated amount of time to pass.
def deallocate(age):
    for tstamp in pipe.results():
        delta = datetime.datetime.now() - tstamp
        duration = datetime.timedelta(seconds=age) - delta
        if duration > datetime.timedelta():
            time.sleep(duration.total_seconds())
        del common[tstamp]
pipe2 = mpipe.Pipeline(mpipe.UnorderedStage(deallocate))
pipe2.put(2)  # Start it up right away.

# Create the OpenCV video capture object.
cap = cv2.VideoCapture(DEVICE)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Run the video capture loop, feeding the image processing pipeline.
now = datetime.datetime.now()
end = now + datetime.timedelta(seconds=DURATION)
while end > now:
    now = datetime.datetime.now()
    hello, image = cap.read()

    # Allocate shared memory for a copy of the input image.
    image_in = sharedmem.empty(np.shape(image), image.dtype)
    
    # Copy the input image to it's shared memory version.
    image_in[:] = image.copy()
    
    # Add image to process-shared table.
    common[now] = {'image_in' : image_in}

    # Feed the pipeline.
    pipe.put(now)

# Signal pipelines to stop, and wait for deallocator
# to free all memory.
pipe.put(None)
pipe2.put(None)
for result in pipe2.results():
    pass

# The end.
