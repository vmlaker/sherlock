"""Multiprocess image processing using shared memory. 
Use frame dropping filter for output viewer."""

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

# Create a process-shared (common) table keyed on timestamps
# and holding references to allocated memory and other useful values.
manager = multiprocessing.Manager()
common = manager.dict()

# Accumulation of thresholded differences.
image_acc = None  

# Keep track of previous iteration's timestamp.
tstamp_prev = None  

def step1(tstamp):
    """Return preprocessed image."""
    global tstamp_prev
    alpha, tstamp_prev = iproc.getAlpha(tstamp_prev, 3)
    
    # Reassign the modified object to the proxy container in order to
    # notify manager that the mutable value (the dictionary) has changed. See for details:
    # http://docs.python.org/library/multiprocessing.html#multiprocessing.managers.SyncManager.list
    itstamp = common[tstamp]
    itstamp['alpha'] = alpha
    common[tstamp] = itstamp  # Reassign modified object to proxy.

    iproc.preprocess(common[tstamp]['image_in'], common[tstamp]['image_pre'])
    return tstamp
 
class Step2Worker(mpipe.OrderedWorker):
    def doTask(self, tstamp):
        """Compute difference between given image and accumulation,
        then accumulate and set result with the difference. 
        Initialize accumulation if needed (if opacity is 100%.)"""
        
        image_pre = common[tstamp]['image_pre']
        alpha = common[tstamp]['alpha']
        
        # Use global accumulation.
        global image_acc

        # Initalize accumulation if so indicated.
        if alpha == 1.0:
            image_acc = np.empty(np.shape(image_pre))

        # Compute difference.
        cv2.absdiff(
            image_acc.astype(image_pre.dtype),
            image_pre,
            common[tstamp]['image_diff']
            )

        self.putResult(tstamp)

        # Accumulate.
        hello = cv2.accumulateWeighted(
            image_pre,
            image_acc,
            alpha,
            )
        
def step3(tstamp):
    """Postprocess image using given difference."""
    iproc.postprocess(
        common[tstamp]['image_in'], 
        common[tstamp]['image_diff'],
        common[tstamp]['image_out'], 
        )
    return tstamp

class Viewer(mpipe.OrderedWorker):
    """Base class viewer implementation, specialized in
    subclasses by overriding getName()."""
    def doTask(self, tstamp):
        name = self.getName()
        try:
            cv2.namedWindow(name, cv2.cv.CV_WINDOW_NORMAL)
            image = common[tstamp][name]
            cv2.imshow(name, image)
            cv2.waitKey(1)
        except:
            print('error running viewer %s !!!'%name)
        return tstamp
    def getName(self): return 'base'

class ViewerIn(Viewer):
    def getName(self): return 'image_in'
class ViewerPre(Viewer):
    def getName(self): return 'image_pre'
class ViewerDiff(Viewer):
    def getName(self): return 'image_diff'
class ViewerOut(Viewer):
    def getName(self): return 'image_out'


def stall(tstamp):
    """Make sure the timestamp is at least a certain 
    age before propagating it further."""
    delta = datetime.datetime.now() - tstamp
    duration = datetime.timedelta(seconds=2) - delta
    if duration > datetime.timedelta():
        time.sleep(duration.total_seconds())
    return tstamp

# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

def printStatus(tstamp):
    """Print the framerate to stdout."""
    print('%05.3f, %05.3f, %05.3f'%framerate.tick())
    return tstamp

# Create the output viewer pipeline.
viewer_out = mpipe.Stage(ViewerOut)
pipe_vout = mpipe.Pipeline(viewer_out)

# Create the diff viewer pipeline.
viewer_diff = mpipe.Stage(ViewerDiff)
pipe_vdiff = mpipe.Pipeline(viewer_diff)

# Create the image processing stages.
step1 = mpipe.OrderedStage(step1)
step2 = mpipe.Stage(Step2Worker)
step3 = mpipe.OrderedStage(step3)

# Create the other, downstream stages.
filter_vdiff = mpipe.FilterStage(pipe_vdiff)
filter_vout = mpipe.FilterStage(pipe_vout)
stall = mpipe.OrderedStage(stall)
printer = mpipe.OrderedStage(printStatus)

# Link the stages into a pipeline.
#
#  step1 ---> step2 ---+---> step3 --------+---> filter_vout ---> stall
#                      |                   |
#                      +---> filter_vdiff  +---> printer
#
step1.link(step2)
step2.link(step3)
step2.link(filter_vdiff)
step3.link(filter_vout)
step3.link(printer)
filter_vout.link(stall)
pipe = mpipe.Pipeline(step1)

# Create an auxiliary pipeline that simply pulls results from
# the image processing pipeline, and deallocates the shared
# memory associated with pulled timestamps. 
def pull(tstamp):
    prev = None
    for tstamp in pipe.results():
        if prev is not None:
            del common[prev]
        prev = tstamp
    del common[prev]
pipe_pull = mpipe.Pipeline(mpipe.UnorderedStage(pull))
pipe_pull.put(True)

# Run the video capture loop, allocating shared memory
# and feeding the image processing pipeline.
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

# Send the "stop" task to all pipelines.
pipe.put(None)
pipe_pull.put(None)
pipe_vdiff.put(None)
pipe_vout.put(None)

# Wait until the pull pipeline processes all it's tasks.
for result in pipe_pull.results():
    pass

# The end.
