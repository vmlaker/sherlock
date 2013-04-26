Sherlock
========

A real-time Python image processing toolkit. 

Sherlock is released as a set of example command-line programs to demonstrate fast image processing of live video (e.g. from a webcam) using symmetric multiprocessing with shared memory, in Python.
It is a collection of test codes. 

Software dependencies
---------------------

Before you can run the codes, have the following libraries installed for your Python interpreter:

### OpenCV

[Open Source Computer Vision](http://opencv.org) is used for image processing (algorithms found in [``cv2``](http://docs.opencv.org/modules/refman.html) module.) 
On a YUM system, install it with:
```
yum install opencv-python
```
or if using Aptitude, try:
```
aptitude install python-opencv
```

### numpy-sharedmem

Images (large NumPy arrays) are efficiently accessed by multiple processes using the ``sharedmem`` module. Take a look at the [numpy-sharedmem project](http://bitbucket.org/cleemesser/numpy-sharedmem) for details.
Install the module with:
```
hg clone https://cleemesser@bitbucket.org/cleemesser/numpy-sharedmem/
cd numpy-sharedmem
python setup.py install --user
```

### MPipe

Multiprocessing workflow is implemented in the [MPipe framework](http://vmlaker.github.io/mpipe/concepts.html). 
Install the module with:
```
git clone http://github.com/vmlaker/mpipe
cd mpipe
python setup.py install --user
```

Usage
-----

Test your OpenCV Python bindings with a simple video playback. The following displays frames captured from the first video device (i.e. ``/dev/video0``) for 5 seconds:
```
python playcv2.py 0 640 480 5
```
Now, run the first test using the following command:
```
python test1.py 0 640 480 5
```
