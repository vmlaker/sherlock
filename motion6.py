"""Multiprocess image processing using shared memory
doing motion detection."""

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
LIFETIMES = map(float, sys.argv[5:])

if not LIFETIMES:
    LIFETIMES = [1.0,]

# Create a process-shared (common) table keyed on timestamps
# and holding references to allocated memory and other useful values.
manager = multiprocessing.Manager()
common = manager.dict()

# Create process-shared tables for each lifetime.
# These hold allocated memory for diff and output images
# corresponding to the particular lifetime.
forked = dict()
for mult in LIFETIMES:
    forked[mult] = manager.dict()


class Step1Worker(mpipe.OrderedWorker):
    """First step of image processing."""
    def doTask(self, tstamp):
        """Return preprocessed image."""
        iproc.preprocess(common[tstamp]['image_in'], common[tstamp]['image_pre'])
        return tstamp
 

class Step2Worker(mpipe.OrderedWorker):
    """Second step of image processing."""

    def __init__(self, lifetime):
        self._lifetime = lifetime  # Alpha age.
        self._image_acc = None  # Accumulation of thresholded differences.
        self._prev_tstamp = None

    def doTask(self, tstamp):
        """Compute difference between given image and accumulation,
        then accumulate and set result with the difference. 
        Initialize accumulation if needed (if opacity is 100%.)"""

        alpha, self._prev_tstamp = iproc.getAlpha(self._prev_tstamp, self._lifetime)
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
        itstamp = forked[self._lifetime][tstamp]
        itstamp['image_diff'] = image_diff
        forked[self._lifetime][tstamp] = itstamp

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

# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

class Step3Worker(mpipe.OrderedWorker):
    """Third step of image processing."""

    def __init__(self, lifetime):
        self._lifetime = lifetime

    def doTask(self, tstamp):
        """Postprocess image using given difference."""

        # Allocate shared memory for the resulting output images.
        image_in = common[tstamp]['image_in']
        shape = np.shape(image_in)
        dtype = image_in.dtype
        image_difft = sharedmem.empty(shape[:2], dtype)
        image_out = sharedmem.empty(shape, dtype)

        # Copy the input image to the output image memory.
        image_out[:] = image_in.copy()

        # Threshold the difference.
        iproc.threshold(
            forked[self._lifetime][tstamp]['image_diff'],
            image_difft,
            )

        # Postprocess the output image.
        # It changes the source image, so pass in a copy.
        iproc.postprocess(
            image_out,
            image_source=image_difft.copy(),
            image_out=image_out,
            )

        # Write the framerate on top of the image.
        iproc.writeOSD(
            image_out, 
            ('%.2f, %.2f, %.2f fps'%framerate.tick(),),
            )

        # Add memory reference to the table.
        # (reassigning modified object to proxy container.)
        itstamp = forked[self._lifetime][tstamp]
        itstamp['image_difft'] = image_difft
        itstamp['image_out'] = image_out
        forked[self._lifetime][tstamp] = itstamp

        return tstamp

class Viewer(mpipe.OrderedWorker):
    """Displays image in a window."""

    def __init__(self, lifetime, image_name):
        """Initialize object with name of image."""
        self._lifetime = lifetime
        self._image_name = image_name
        self._win_name = '%s  (lifetime=%ss)'%(self._image_name, self._lifetime)

    def doInit(self):
        """Run namedWindow() in the viewer worker process."""
        cv2.namedWindow(self._win_name, cv2.cv.CV_WINDOW_NORMAL)

    def doTask(self, tstamp):
        try:
            image = forked[self._lifetime][tstamp][self._image_name]
            cv2.imshow(self._win_name, image)
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

# Create the starting image processing stage.
step1 = mpipe.Stage(Step1Worker)

# Create the remainder of downstream stages.
view_pipes = list()  # Keep a list of viewer sub-pipes.
for age in LIFETIMES:

    # Create the downstream image processing stages.
    step2 = mpipe.Stage(Step2Worker, size=1, lifetime=age)
    step3 = mpipe.Stage(Step3Worker, size=1, lifetime=age)

    # Create the two viewer pipelines. These will be "wrapped" 
    # in frame-dropping filters to eliminate bottlenecks.
    pipe_vout = mpipe.Pipeline(
        mpipe.Stage(Viewer, 1, lifetime=age, image_name='image_out'))
    pipe_vdiff = mpipe.Pipeline(
        mpipe.Stage(Viewer, 1, lifetime=age, image_name='image_diff'))
    pipe_vdifft = mpipe.Pipeline(
        mpipe.Stage(Viewer, 1, lifetime=age, image_name='image_difft'))
    view_pipes.append(pipe_vout)
    view_pipes.append(pipe_vdiff)
    view_pipes.append(pipe_vdifft)

    # Create the other stages, further downstream.
    filter_v1 = mpipe.FilterStage((pipe_vdiff,), drop_results=True)
    filter_v2 = mpipe.FilterStage((pipe_vout,pipe_vdifft,), drop_results=True)
    stall = mpipe.Stage(Staller)

    # Link the stages together.
    # This is the sub-pipe, downstream of the first stage.
    #
    #                           (difft viewer)
    #                            (out viewer)
    #                                 ||
    #    step2 ---+--> step3 ---> filter_v2 ---> stall
    #             |                  
    #             +--> filter_v1
    #                      ||
    #                (diff viewer)
    #
    step2.link(step3)
    step2.link(filter_v1)
    step3.link(filter_v2)
    filter_v2.link(stall)

    # Now link the sub-pipe above with the first stage.
    #
    #    step1 ---> step2 ...
    #
    step1.link(step2)

# Finally, create the image processing pipeline object.
pipe_iproc = mpipe.Pipeline(step1)

# Create an auxiliary process (modeled as a one-task pipeline)
# that simply pulls results from the image processing pipeline, 
# and deallocates the associated shared memory.
def pull(task):
    for tstamp in pipe_iproc.results():
        del common[tstamp]
        for age in LIFETIMES:
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
    # for the different lifetimes.
    for age in LIFETIMES:
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
