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
            common[tstamp]['image_diff']
            )

        self.putResult(tstamp)

        # Accumulate.
        hello = cv2.accumulateWeighted(
            image_pre,
            self._image_acc,
            alpha * self._alpha_mult,
            )
        
# Monitor framerates for the given seconds past.
framerate = util.RateTicker((1,5,10))

def step3(tstamp):
    """Postprocess image using given difference."""
    iproc.postprocess(
        common[tstamp]['image_in'], 
        common[tstamp]['image_diff'],
        common[tstamp]['image_out'], 
        )

    # Write the framerate on top of the image.
    iproc.writeOSD(
        common[tstamp]['image_out'],
        ('%.2f, %.2f, %.2f fps'%framerate.tick(),),
        ratio=0.04,
        )

    return tstamp

class Viewer(mpipe.OrderedWorker):
    """Displays image in a window."""

    def __init__(self, image_name):
        """Initialize object with name of image."""
        self._image_name = image_name

    def doTask(self, tstamp):
        try:
            cv2.namedWindow(self._image_name, cv2.cv.CV_WINDOW_NORMAL)
            image = common[tstamp][self._image_name]
            cv2.imshow(self._image_name, image)
            cv2.waitKey(1)
        except:
            print('error running viewer %s !!!'%self._image_name)
        return tstamp


def stall(tstamp):
    """Make sure the timestamp is at least a certain 
    age before propagating it further."""
    delta = datetime.datetime.now() - tstamp
    duration = datetime.timedelta(seconds=2) - delta
    if duration > datetime.timedelta():
        time.sleep(duration.total_seconds())
    return tstamp


# Create the two viewer pipelines.
pipe_vout = mpipe.Pipeline(
    mpipe.Stage(Viewer, 1, image_name='image_out'))
pipe_vdiff = mpipe.Pipeline(
    mpipe.Stage(Viewer, 1, image_name='image_diff'))

# Create the image processing stages.
step1 = mpipe.OrderedStage(step1)
step2 = mpipe.Stage(Step2Worker, size=1, alpha_mult=0.50)
step3 = mpipe.OrderedStage(step3)

# Create the other, downstream stages.
filter_vdiff = mpipe.FilterStage((pipe_vdiff,), drop_results=True)
filter_vout = mpipe.FilterStage((pipe_vout,), drop_results=True)
stall = mpipe.OrderedStage(stall)

# Link the stages into the image processing pipeline:
#
#  step1 ---> step2 --+--> step3 --------+--> filter_vout ---> stall
#                     |                 
#                     +--> filter_vdiff 
#
step1.link(step2)
step2.link(step3)
step2.link(filter_vdiff)
step3.link(filter_vout)
filter_vout.link(stall)
pipe_iproc = mpipe.Pipeline(step1)

# Create an auxiliary pipeline that simply pulls results from
# the image processing pipeline, and deallocates the shared
# memory associated with pulled timestamps. 
def pull(tstamp):
    prev = None
    for tstamp in pipe_iproc.results():
        if prev is not None:
            del common[prev]
        prev = tstamp
    del common[prev]
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
    #   the preprocessed image,
    #   the diff image,
    #   the resulting output image.
    shape = np.shape(image)
    dtype = image.dtype
    image_in   = sharedmem.empty(shape,     dtype)
    image_pre  = sharedmem.empty(shape[:2], dtype)
    image_diff = sharedmem.empty(shape[:2], dtype)
    image_out  = sharedmem.empty(shape,     dtype)
    
    # Copy the input image to its shared memory version,
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
    pipe_iproc.put(now)

# Send the "stop" task to all pipelines.
pipe_iproc.put(None)
pipe_pull.put(None)
pipe_vdiff.put(None)
pipe_vout.put(None)

# Wait until the pull pipeline processes all it's tasks.
for result in pipe_pull.results():
    pass

# The end.
