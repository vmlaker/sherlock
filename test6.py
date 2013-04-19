"""Multiprocess image processing using shared memory. 
Use frame dropping filter for output viewer.
Use more than one alpha multiplier."""

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
ALPHA_MULTS = map(float, sys.argv[5:])

if not ALPHA_MULTS:
    ALPHA_MULTS = [1.0,]

# Create the OpenCV video capture object.
cap = cv2.VideoCapture(DEVICE)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Create a process-shared (common) table keyed on timestamps
# and holding references to allocated memory and other useful values.
manager = multiprocessing.Manager()
common = manager.dict()

# Create process-shared tables for each alpha multiplier.
# These hold allocated memory for diff and output images
# corresponding to the particular alpha multiplier.
forked = dict()
for mult in ALPHA_MULTS:
    forked[mult] = manager.dict()

# Keep track of previous iteration's timestamp.
tstamp_prev = None  

def step1(tstamp):
    """Return preprocessed image."""
    global tstamp_prev
    alpha, tstamp_prev = iproc.getAlpha(tstamp_prev, 1)
    
    # Reassign the modified object to the proxy container in order to
    # notify manager that the mutable value (the dictionary) has changed. See for details:
    # http://docs.python.org/library/multiprocessing.html#multiprocessing.managers.SyncManager.list
    itstamp = common[tstamp]
    itstamp['alpha'] = alpha
    common[tstamp] = itstamp  # Reassign modified object to proxy.

    iproc.preprocess(common[tstamp]['image_in'], common[tstamp]['image_pre'])
    return tstamp
 

class Step2Worker(mpipe.OrderedWorker):

    def __init__(self, alpha_mult):
        self._alpha_mult = alpha_mult  # Alpha multiplier.
        self._image_acc = None  # Accumulation of thresholded differences.

    def doTask(self, tstamp):
        """Compute difference between given image and accumulation,
        then accumulate and set result with the difference. 
        Initialize accumulation if needed (if opacity is 100%.)"""
        
        image_pre = common[tstamp]['image_pre']
        alpha = common[tstamp]['alpha']
        
        # Initalize accumulation if so indicated.
        if alpha == 1.0:
            self._image_acc = np.empty(np.shape(image_pre))

        # Compute difference.
        cv2.absdiff(
            self._image_acc.astype(image_pre.dtype),
            image_pre,
            forked[self._alpha_mult][tstamp]['image_diff'],
            )

        self.putResult(tstamp)

        # Accumulate.
        hello = cv2.accumulateWeighted(
            image_pre,
            self._image_acc,
            alpha * self._alpha_mult,
            )

class Step3Worker(mpipe.OrderedWorker):

    def __init__(self, alpha_mult):
        self._alpha_mult = alpha_mult  # Alpha multiplier.

    def doTask(self, tstamp):
        """Postprocess image using given difference."""
        iproc.postprocess(
            forked[self._alpha_mult][tstamp]['image_out'], 
            forked[self._alpha_mult][tstamp]['image_diff'],
            forked[self._alpha_mult][tstamp]['image_out'], 
            )
        return tstamp

class Viewer(mpipe.OrderedWorker):
    """Displays image in a window."""

    def __init__(self, alpha_mult, image_name):
        """Initialize object with name of image."""
        self._alpha_mult = alpha_mult  # Alpha multiplier.
        self._image_name = image_name

    def doTask(self, tstamp):
        try:
            win_name = '%s-%s'%(self._image_name, self._alpha_mult)
            cv2.namedWindow(win_name, cv2.cv.CV_WINDOW_NORMAL)
            image = forked[self._alpha_mult][tstamp][self._image_name]
            cv2.imshow(win_name, image)
            cv2.waitKey(1)
        except:
            print('error running viewer %s !!!'%self._image_name)
        return tstamp


class Staller(mpipe.OrderedWorker):
    def doTask(self, tstamp):
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

# Create the first image processing stage 
# and the downstream printer.
step1 = mpipe.OrderedStage(step1)
printer = mpipe.OrderedStage(printStatus)

# Link the first stages:
#
#    step1 ---> printer
#
step1.link(printer)

view_pipes = list()

for mult in ALPHA_MULTS:

    # Create the two viewer pipelines.
    pipe_vout = mpipe.Pipeline(
        mpipe.Stage(Viewer, 1, alpha_mult=mult, image_name='image_out'))
    pipe_vdiff = mpipe.Pipeline(
        mpipe.Stage(Viewer, 1, alpha_mult=mult, image_name='image_diff'))
    view_pipes.append(pipe_vout)
    view_pipes.append(pipe_vdiff)

    # Create the downstream image processing stages.
    step2 = mpipe.Stage(Step2Worker, size=1, alpha_mult=mult)
    step3 = mpipe.Stage(Step3Worker, size=1, alpha_mult=mult)

    # Create the other, downstream stages.
    filter_vdiff = mpipe.FilterStage(pipe_vdiff)
    filter_vout = mpipe.FilterStage(pipe_vout)
    stall = mpipe.Stage(Staller)

    # Link the stages together:
    #
    #    step2 ---+--> step3 ---> filter_vout ---> stall
    #             |                  
    #             +--> filter_vdiff  
    #
    step2.link(step3)
    step2.link(filter_vdiff)
    step3.link(filter_vout)
    filter_vout.link(stall)

    # Link the chain of stages to the first image processing stage:
    #
    #    step1 ---> step2 ---> ...
    #
    step1.link(step2)

# Finally, create the image processing pipeline.
pipe_iproc = mpipe.Pipeline(step1)

# Create an auxiliary pipeline that simply pulls results from
# the image processing pipeline, and deallocates the shared
# memory associated with pulled timestamps. 
def pull(tstamp):
    prev = None
    for tstamp in pipe_iproc.results():
        if prev is not None:
            del common[prev]
            for mult in ALPHA_MULTS:
                del forked[mult][prev]
        prev = tstamp
    del common[prev]
    for mult in ALPHA_MULTS:
        del forked[mult][prev]
pipe_pull = mpipe.Pipeline(mpipe.UnorderedStage(pull))
pipe_pull.put(True)  # Start it up right away.

# Run the video capture loop, allocating shared memory
# and feeding the image processing pipeline.
now = datetime.datetime.now()
end = now + datetime.timedelta(seconds=DURATION)
while end > now:
    now = datetime.datetime.now()
    hello, image = cap.read()

    # Allocate shared memory for
    #   a copy of the input image,
    #   the preprocessed image.
    shape = np.shape(image)
    dtype = image.dtype
    image_in   = sharedmem.empty(shape,     dtype)
    image_pre  = sharedmem.empty(shape[:2], dtype)
    
    # Copy the input image to its shared memory version.
    image_in[:] = image.copy()
    
    # Add shared memory (references) to the common table.
    common[now] = {
        'image_in'   : image_in,
        'image_pre'  : image_pre,
        'alpha'      : 1.0,
        }

    # Allocate memory for the different alpha multipliers,
    # and add their references to the tables.
    for mult in ALPHA_MULTS:
        # Allocate shared memory for
        #   the diff image,
        #   the resulting output image.
        image_diff = sharedmem.empty(shape[:2], dtype)
        image_out  = sharedmem.empty(shape,     dtype)

        # Copy the input image to the eventual output image memory.
        image_out[:] = image.copy()

        # Add memory references to the table.
        forked[mult][now] = {
            'image_diff' : image_diff,
            'image_out'  : image_out,
            }

    # Put the timestamp on the image processing pipeline.
    pipe_iproc.put(now)

# Send the "stop" task to all pipelines.
pipe_iproc.put(None)
pipe_pull.put(None)
for pipe in view_pipes:
    pipe.put(None)

# Wait until the pull pipeline processes all it's tasks.
for result in pipe_pull.results():
    pass

# The end.
