.. image:: http://vmlaker.github.io/sherlock/logo.png
  :alt: Sherlock Logo

Sherlock
========

*OpenCV in Python, with multiprocessing and shared memory!*

A collection of small codes for processing live video 
(like from a webcam) with `OpenCV <http://opencv.org>`_.
Image data in memory (NumPy arrays) are shared by multiple processes using
`numpy-sharedmem Python module <http://bitbucket.org/cleemesser/numpy-sharedmem>`_.
Parallel processing workflow is implemented with 
`MPipe <http://vmlaker.github.io/mpipe/concepts.html>`_. 

Download and Install
--------------------

#. First thing, you'll need OpenCV Python bindings on your system.
   Get it using **YUM**:
   ::

      yum install opencv-python

   or **Aptitude**:
   ::

      aptitude install python-opencv

   or **Homebrew** (on OS X):
   ::
   
      brew tap homebrew/science
      brew install opencv

#. Okay, now get the project code:
   ::

      git clone --recursive http://github.com/vmlaker/sherlock

#. Run make:
   ::

      cd sherlock
      make

Playback test
-------------

For starters, let's test your OpenCV Python bindings.
This command shows live view from the first video device 
(i.e. ``/dev/video0``) for a duration of 8 seconds:
::

   make playcv2

Motion detection
----------------

The following cases demonstrate simplified motion detection.
Each iteraton increases in complexity with cumulative changes 
intended to enhance performance. 

#. Run one process:
   ::

      make diffavg1

#. Add parallel processing:
   ::
      
      make diffavg2

#. Add shared memory:
   ::

      make diffavg3

#. Add filtering:
   ::

      make diffavg4

Processing algorithm is a run-of-the-mill foreground/background segmentation using scene average. 
Consider profiling resource usage by running with ``time`` command.

Object detection
----------------

Objects in the video stream are detected using Haar feature-based 
cascade classifiers. Active classifiers are listed in
``src/util/cascade.py`` file. By default, these are 
vanilla classifiers shipped with OpenCV distribution.
You can edit this file to activate (or deactivate) classifiers,
change search paths, add your own custom classifiers,
and configure global object detection parameters.

Run face detection serially:
::

   make object1

Run face detection in parallel:
::

   make object2
