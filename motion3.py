"""Multiprocess motion detection using shared memory."""

import multiprocessing
import datetime
import time
import sys
import cv2
import numpy as np

import sharedmem
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
cv2.namedWindow('motion detection 3', cv2.cv.CV_WINDOW_NORMAL)

# Create a process-shared (common) table keyed on timestamps
# and holding references to allocated memory and other useful values.
manager = multiprocessing.Manager()
common = manager.dict()

# Maintain accumulation of thresholded differences.
image_acc = None  

# Keep track of previous iteration's timestamp.
tstamp_prev = None  

def step1(tstamp):
    """Return preprocessed image."""
    global tstamp_prev
    alpha, tstamp_prev = iproc.getAlpha(tstamp_prev)
    
    # Reassign the modified object to the proxy container in order to
    # notify manager that the mutable value (the dictionary) has changed. See for details:
    # http://docs.python.org/library/multiprocessing.html#multiprocessing.managers.SyncManager.list
    itstamp = common[tstamp]
    itstamp['alpha'] = alpha
    common[tstamp] = itstamp  # Reassign modified object to proxy.

    iproc.preprocess(common[tstamp]['image_in'], common[tstamp]['image_pre'])
    return tstamp
 
def step2(tstamp):
    """Compute difference between given image and accumulation,
    then accumulate and set result with the difference. 
    Initialize accumulation if needed (if opacity is 100%.)"""
        
    image_pre = common[tstamp]['image_pre']
    image = common[tstamp]['image_in']
    alpha = common[tstamp]['alpha']
        
    # Initalize accumulation if so indicated.
    global image_acc
    if image_acc is None:
        image_acc = np.empty(np.shape(image_pre))

    # Compute difference.
    cv2.absdiff(
        image_acc.astype(image_pre.dtype),
        image_pre,
        common[tstamp]['image_diff']
        )

    # Accumulate.
    hello = cv2.accumulateWeighted(
        image_pre,
        image_acc,
        alpha,
        )
    return tstamp

# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

def step3(tstamp):
    """Postprocess image using given difference."""
    image_difft = iproc.threshold(common[tstamp]['image_diff'])
    iproc.postprocess(
        common[tstamp]['image_in'], 
        image_source=image_difft,
        image_out=common[tstamp]['image_out'], 
        )

    # Write the framerate on top of the image.
    iproc.writeOSD(
        common[tstamp]['image_out'],
        ('%.2f, %.2f, %.2f fps'%framerate.tick(),),
        )

    return tstamp

def step4(tstamp):
    """Display the output image."""
    cv2.imshow('motion detection 3', common[tstamp]['image_out'])
    cv2.waitKey(1)  # Allow HighGUI to process event.
    return tstamp

def step5(tstamp):
    """Make sure the timestamp is at least a certain 
    age before propagating it further."""
    delta = datetime.datetime.now() - tstamp
    duration = datetime.timedelta(seconds=2) - delta
    if duration > datetime.timedelta():
        time.sleep(duration.total_seconds())
    return tstamp

stage1 = mpipe.OrderedStage(step1)
stage2 = mpipe.OrderedStage(step2)
stage3 = mpipe.OrderedStage(step3)
stage4 = mpipe.OrderedStage(step4)
stage5 = mpipe.OrderedStage(step5)
stage1.link(stage2)
stage2.link(stage3)
stage3.link(stage4)
stage4.link(stage5)
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

    # Allocate shared memory for
    #   a copy of the input image,
    #   the preprocessed image,
    #   the diff image,
    #   the resulting output image.
    shape = np.shape(image)
    dtype = image.dtype
    image_in   = sharedmem.empty(shape,     dtype)
    image_pre  = sharedmem.empty(shape[:2], dtype)
    image_diff = sharedmem.empty(shape[:2], dtype)
    image_out  = sharedmem.empty(shape,     dtype)
    
    # Copy the input image to it's shared memory version,
    # and also to the eventual output image memory.
    image_in[:] = image.copy()
    image_out[:] = image.copy()
    
    common[now] = {
        'image_in'   : image_in,
        'image_pre'  : image_pre,
        'image_diff' : image_diff,
        'image_out'  : image_out,
        'alpha'      : 1.0,
        }
    pipe.put(now)

# Signal processing pipeline to stop.
pipe.put(None)

# Signal deallocator to stop and wait until it frees all memory.
pipe2.put(None)
for result in pipe2.results():
    pass

# The end.
