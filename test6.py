"""Multiprocess image processing using shared memory. 
Use frame dropping filter for output viewer.
Use more than one alpha age."""

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
ALPHA_AGES = map(float, sys.argv[5:])  # Values in range [0.0, 1.0].

if not ALPHA_AGES:
    ALPHA_AGES = [1.0,]

# Create a process-shared (common) table keyed on timestamps
# and holding references to allocated memory and other useful values.
manager = multiprocessing.Manager()
common = manager.dict()

# Create process-shared tables for each alpha age.
# These hold allocated memory for diff and output images
# corresponding to the particular alpha age.
forked = dict()
for mult in ALPHA_AGES:
    forked[mult] = manager.dict()


class Step1Worker(mpipe.OrderedWorker):
    """First step of image processing."""
    def doTask(self, tstamp):
        """Return preprocessed image."""
        iproc.preprocess(common[tstamp]['image_in'], common[tstamp]['image_pre'])
        return tstamp
 

class Step2Worker(mpipe.OrderedWorker):
    """Second step of image processing."""

    def __init__(self, alpha_age):
        self._alpha_age = alpha_age  # Alpha age.
        self._image_acc = None  # Accumulation of thresholded differences.
        self._prev_tstamp = None

    def doTask(self, tstamp):
        """Compute difference between given image and accumulation,
        then accumulate and set result with the difference. 
        Initialize accumulation if needed (if opacity is 100%.)"""

        alpha, self._prev_tstamp = iproc.getAlpha(self._prev_tstamp, self._alpha_age)
        image_pre = common[tstamp]['image_pre']
        
        # Allocate shared memory for the diff image.
        image_in = common[tstamp]['image_in']
        shape = np.shape(image_in)
        dtype = image_in.dtype
        image_diff = sharedmem.empty(shape[:2], dtype)

        # Add memory references to the table
        # Reassign the modified object to the proxy container in order to
        # notify manager that the mutable value (the dictionary) has changed. See for details:
        # http://docs.python.org/library/multiprocessing.html#multiprocessing.managers.SyncManager.list
        itstamp = forked[self._alpha_age][tstamp]
        itstamp['image_diff'] = image_diff
        forked[self._alpha_age][tstamp] = itstamp

        # Initalize accumulation if so indicated.
        if self._image_acc is None:
            self._image_acc = np.empty(np.shape(image_pre))

        # Compute difference.
        cv2.absdiff(
            self._image_acc.astype(image_pre.dtype),
            image_pre,
            image_diff,
            )

        self.putResult(tstamp)

        # Accumulate.
        hello = cv2.accumulateWeighted(
            image_pre,
            self._image_acc,
            alpha,
            )


class Step3Worker(mpipe.OrderedWorker):
    """Third step of image processing."""

    def __init__(self, alpha_age):
        self._alpha_age = alpha_age  # Alpha age.

    def doTask(self, tstamp):
        """Postprocess image using given difference."""

        # Allocate shared memory for the resulting output image.
        image_in = common[tstamp]['image_in']
        shape = np.shape(image_in)
        dtype = image_in.dtype
        image_out = sharedmem.empty(shape, dtype)

        # Copy the input image to the output image memory.
        image_out[:] = image_in.copy()

        # Add memory reference to the table.
        # (reassigning modified object to proxy container.)
        itstamp = forked[self._alpha_age][tstamp]
        itstamp['image_out'] = image_out
        forked[self._alpha_age][tstamp] = itstamp

        # Postprocess the output image.
        iproc.postprocess(
            image_out,
            forked[self._alpha_age][tstamp]['image_diff'],
            image_out,
            )
        return tstamp

class Viewer(mpipe.OrderedWorker):
    """Displays image in a window."""

    def __init__(self, alpha_age, image_name):
        """Initialize object with name of image."""
        self._alpha_age = alpha_age  # Alpha age.
        self._image_name = image_name

    def doTask(self, tstamp):
        try:
            win_name = '%s  (alpha age=%ss)'%(self._image_name, self._alpha_age)
            cv2.namedWindow(win_name, cv2.cv.CV_WINDOW_NORMAL)
            image = forked[self._alpha_age][tstamp][self._image_name]
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

# Create the starting image processing stage, and the single
# downstream printer, and link them up:
#
#    step1 ---> printer
#
step1 = mpipe.Stage(Step1Worker)
printer = mpipe.OrderedStage(printStatus)
step1.link(printer)

# Create the 
view_pipes = list()  # Keep a list of viewer sub-pipes.
for age in ALPHA_AGES:

    # Create the downstream image processing stages.
    step2 = mpipe.Stage(Step2Worker, size=1, alpha_age=age)
    step3 = mpipe.Stage(Step3Worker, size=1, alpha_age=age)

    # Create the two viewer pipelines. These will be "wrapped" 
    # in frame-dropping filters to eliminate bottlenecks.
    pipe_vout = mpipe.Pipeline(
        mpipe.Stage(Viewer, 1, alpha_age=age, image_name='image_out'))
    pipe_vdiff = mpipe.Pipeline(
        mpipe.Stage(Viewer, 1, alpha_age=age, image_name='image_diff'))
    view_pipes.append(pipe_vout)
    view_pipes.append(pipe_vdiff)

    # Create the other stages, further downstream.
    filter_vdiff = mpipe.FilterStage((pipe_vdiff,), drop_results=True)
    filter_vout = mpipe.FilterStage((pipe_vout,), drop_results=True)
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

# Create an auxiliary process (modeled as a one-task pipeline)
# that simply pulls results from the image processing pipeline, 
# and deallocates the associated shared memory.
def pull(task):
    for tstamp in pipe_iproc.results():
        del common[tstamp]
        for age in ALPHA_AGES:
            del forked[age][tstamp]
pipe_pull = mpipe.Pipeline(mpipe.UnorderedStage(pull))
pipe_pull.put(True)  # Start it up right away.

# Create the OpenCV video capture object.
cap = cv2.VideoCapture(DEVICE)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Run the video capture loop, allocating shared memory
# and feeding the image processing pipeline.
now = datetime.datetime.now()
end = now + datetime.timedelta(seconds=DURATION)
while end > now:

    # Mark the timestamp. This is the index by which 
    # image procesing stages will access allocated memory.
    now = datetime.datetime.now()

    # Capture the image.
    hello, image = cap.read()

    # Allocate shared memory for a copy of the input image,
    # and for the preprocessed image.
    shape = np.shape(image)
    dtype = image.dtype
    image_in   = sharedmem.empty(shape,     dtype)
    image_pre  = sharedmem.empty(shape[:2], dtype)
    
    # Copy the input image to its shared memory version.
    image_in[:] = image.copy()
    
    # Add shared memory (references) to the common table.
    common[now] = {
        'image_in'   : image_in,   # Input image.
        'image_pre'  : image_pre,  # Preprocessed image.
        }

    # Initialize dictionaries that will hold allocated memory
    # for the different alpha ages.
    for age in ALPHA_AGES:
        forked[age][now] = dict()

    # Put the timestamp on the image processing pipeline.
    pipe_iproc.put(now)

# Capturing of video is done. Now let's shut down all
# pipelines by sending them the "stop" task.
pipe_iproc.put(None)
pipe_pull.put(None)
for pipe in view_pipes:
    pipe.put(None)

# Before exiting, wait until the pull pipeline is done
# so that all shared memory is deallocated.
for result in pipe_pull.results():
    pass

# The end.
