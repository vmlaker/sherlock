"""Full multiple process image processing -- with shared memory
and GStreamer webcam streaming."""

import multiprocessing
import sharedmem
import numpy
import cv2

import datetime
import sys

import mpipe

import util
import iproc

DEVICE   = int(sys.argv[1])
WIDTH    = int(sys.argv[2])
HEIGHT   = int(sys.argv[3])
DURATION = float(sys.argv[4])

# Create process-shared tables, 
# one holding allocated memories keyed on timestamp,
# another holding other common process-shared values.
manager = multiprocessing.Manager()
memories = manager.dict()
common = manager.dict()

# Initialize the common accumulated image to None, 
# indicating that analysis hasn't started yet.
common['image_acc'] = None


class Allocator(mpipe.OrderedWorker):
    """Allocates shared memory to be used in the remainder of pipeline."""
    def doTask(self, task):
        try:
            tstamp = datetime.datetime.now()

            # Allocate shared memory for
            #   a copy of the input image,
            #   the preprocessed image,
            #   the diff image,
            #   the postprocessed image,
            #   the resulting output image.
            shape = numpy.shape(task['image_in'])
            dtype = task['image_in'].dtype
            image_in   = sharedmem.empty(shape,     dtype)
            image_pre  = sharedmem.empty(shape[:2], dtype)
            image_diff = sharedmem.empty(shape[:2], dtype)
            image_post = sharedmem.empty(shape[:2], dtype)
            image_out  = sharedmem.empty(shape,     dtype)

            # Copy the input image to it's shared memory version,
            # and also to the eventual output image memory.
            image_in[:] = task['image_in'].copy()
            image_out[:] = task['image_in'].copy()

            # Store all allocated memory in the table because
            # we will later explicitly deallocate the memory.
            index = task['tstamp_onbuffer1']
            memories[index] = (
                image_in, 
                image_pre, 
                image_diff, 
                image_post,
                image_out,
                )

            # Retrieve the common accumulated image.
            image_acc = common['image_acc']

            # If this is the first task through the pipeline 
            # (indicated by the accumulated image being None),
            # then allocate shared memory for accumulated image
            # and set the accumulation alpha value to 100%.
            if image_acc is None:
                # Allocate shared memory for the accumulator image.
                image_acc = sharedmem.zeros(numpy.shape(image_pre))
                cv2.accumulateWeighted(image_pre, image_acc, 1.000)
                common['image_acc'] = image_acc
                alpha = 1.0  # Initially transparency is zero.

            # Otherwise compute accumulation alpha value based 
            # on time elapsed since the previous task.
            else:
                tdelta = task['tstamp_onbuffer1'] - common['prev_tstamp']
                alpha = tdelta.total_seconds()
                alpha *= 0.50  # 1/2.

            common['prev_tstamp'] = task['tstamp_onbuffer1']

            # Prepare the task for the next stage.
            task['image_in'] = image_in
            task['image_pre'] = image_pre
            task['image_diff'] = image_diff
            task['image_post'] = image_post
            task['image_out'] = image_out
            task['alpha'] = alpha

            task['tstamp_alloc1'] = tstamp
            task['tstamp_alloc2'] = datetime.datetime.now()

        except:
            print('error running allocator !!!')

        self.putResult(task)


def step1(task):
    """Return preprocessed image."""
    tstamp = datetime.datetime.now()
    iproc.preprocess(task['image_in'], task['image_pre'])
    task['tstamp_pre1'] = tstamp
    task['tstamp_pre2'] = datetime.datetime.now()
    return task


class DiffAndAccumulator(mpipe.OrderedWorker):
    """Performs diff and accumulation."""
    def doTask(self, task):
        try:
            tstamp = datetime.datetime.now()

            # Compute the difference.
            cv2.absdiff(
                common['image_acc'].astype(task['image_pre'].dtype), 
                task['image_pre'],
                task['image_diff'],
                )

            task['tstamp_diff1'] = tstamp
            task['tstamp_diff2'] = datetime.datetime.now()

            # Allow the next stage in the pipeline to begin work on the result.
            self.putResult(task)

            # Accumulate images.
            hello = cv2.accumulateWeighted(
                task['image_pre'],
                common['image_acc'],
                task['alpha'],
                )
        except:
            print('error running accumulator !!!')


def step3(task):
    """Pipeline element that augments original image."""
    tstamp = datetime.datetime.now()
    iproc.postprocess(
        task['image_post'],
        task['image_diff'],
        task['image_out'],
        )
    task['tstamp_post1'] = tstamp
    task['tstamp_post2'] = datetime.datetime.now()
    return task


class Printer(mpipe.OrderedWorker):
    """Prints some useful info to standard output."""
    def doTask(self, task):
        try:
            task['tstamp_print1'] = datetime.datetime.now()

            tworkers = 0.0
            tgaps = 0.0

            # Assemble the timings string.
            timings = ''
            specs = (
                ('tstamp_onbuffer1', 'tstamp_onbuffer2'),
                ('tstamp_onbuffer2', 'tstamp_alloc1'),
                ('tstamp_alloc1',    'tstamp_alloc2'),
                ('tstamp_alloc2',    'tstamp_pre1'),
                ('tstamp_pre1',      'tstamp_pre2'),
                ('tstamp_pre2',      'tstamp_diff1'),
                ('tstamp_diff1',     'tstamp_diff2'),
                ('tstamp_diff2',     'tstamp_post1'),
                ('tstamp_post1',     'tstamp_post2'),
                ('tstamp_post2',     'tstamp_print1'),
                )
            for element in specs:
                elapsed = -(task[element[0]] - task[ element[1]]).total_seconds()

                # Even indexes are for work in stages.
                if specs.index(element)%2 == 0:
                    timings += '(%0.3f) '%elapsed
                    tworkers += elapsed
                # Odd indexes are for gaps (i.e. communication) between stages.
                else:
                    timings += '%0.3f '%elapsed
                    tgaps += elapsed

            # Dump to standard output.
            print('%s  elapsed= %0.3f (%0.3f + %0.3f) pipe= %s fps= %s'%(
                task['tstamp_onbuffer1'], 
                tworkers + tgaps,
                tworkers,
                tgaps,
                timings,
                task['fps_onbuffer'],
                ))

        except:
            print('error running printer !!!')

        return task


class Viewer(mpipe.OrderedWorker):
    def doTask(self, task):
        try:
            name = self.getName()
            cv2.namedWindow(name, cv2.cv.CV_WINDOW_NORMAL)
            cv2.imshow(name, task[name])
            cv2.waitKey(1)
        except:
            print('error running viewer %s !!!'%self.getName())
        return task

class ViewerIn(Viewer):
    def getName(self):
        return 'image_in'
class ViewerPre(Viewer):
    def getName(self):
        return 'image_pre'
class ViewerDiff(Viewer):
    def getName(self):
        return 'image_diff'
class ViewerPost(Viewer):
    def getName(self):
        return 'image_post'
class ViewerOut(Viewer):
    def getName(self):
        return 'image_out'


class ImageProcessor(mpipe.Pipeline):
    """Deletes shared memory upon get()."""
    def get(self):
        result = super(ImageProcessor, self).get()
        if result is None:
            return None
        
        # Deleting elements from the inter-process (shared) dictionary
        # eventually leads to deallocation of shared memory: 
        # once the dictionary is synced on the process that owns the memory, 
        # the owning process loses reference to the memory and 
        # deallocation is delegated to the garbage collector.
        index = result['tstamp_onbuffer1']
        del memories[index]

        return result


# Create stages of for the image processing pipeline.
stages = list()
stages.append(mpipe.Stage(Allocator))
stages.append(mpipe.OrderedStage(step1))
stages.append(mpipe.Stage(DiffAndAccumulator))
stages.append(mpipe.OrderedStage(step3))
stages.append(mpipe.Stage(Printer))
stages.append(mpipe.Stage(ViewerIn))
stages.append(mpipe.Stage(ViewerPre))
stages.append(mpipe.Stage(ViewerDiff))
stages.append(mpipe.Stage(ViewerPost))
stages.append(mpipe.Stage(ViewerOut))

# Link the stages.
stages[0].link(stages[1])
stages[1].link(stages[2])
stages[2].link(stages[3])
stages[3].link(stages[4])
stages[0].link(stages[5])
#stages[1].link(stages[6])
#stages[2].link(stages[7])
#stages[3].link(stages[8])
stages[3].link(stages[9])

# Create the image processor.
iprocessor = ImageProcessor(stages[0])

def pull(tstamp):
    for tstamp in iprocessor.results():
        pass
pipe2 = mpipe.Pipeline(mpipe.UnorderedStage(pull))
pipe2.put(True)
pipe2.put(None)

# Create the OpenCV video capture object.
cap = cv2.VideoCapture(DEVICE)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# Run the capture loop.
now = datetime.datetime.now()
end = now + datetime.timedelta(seconds=DURATION)
rticker = util.RateTicker((1,5,10))
while end > now:
    now = datetime.datetime.now()
    hello, image = cap.read()
    task = {}
    task['image_in'] = image
    task['fps_onbuffer'] = '%05.3f, %05.3f, %05.3f'%rticker.tick()
    task['tstamp_onbuffer1'] = now
    task['tstamp_onbuffer2'] = datetime.datetime.now()
    iprocessor.put(task)

# Shutdown all the pieces.
iprocessor.put(None)
for result in pipe2.results():
    pass

# The end.
