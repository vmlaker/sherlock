Sherlock
========

A realtime image processing toolkit in Python. 

Sherlock is released as a set of example command-line Python programs which demonstrate fast image processing functionality of [OpenCV](http://opencv.org). It is a collection of test codes you see there in the root directory. Before you can run the example programs however, you gotta have the following libraries installed for your Python interpreter:

* [OpenCV](http://opencv.org)
* [numpy-sharedmem](http://bitbucket.org/cleemesser/numpy-sharedmem)
* [MPipe](http://vmlaker.github.io/mpipe)

Image processing algorithms used are found in OpenCV's [cv2 module](http://docs.opencv.org/modules/refman.html). For sharing NumPy arrays accross processors, [numpy-sharedmem](http://bitbucket.org/cleemesser/numpy-sharedmem) module is doing all the work. Multiprocessing workflow is implemented in the [MPipe pipeline framework](http://vmlaker.github.io/mpipe/concepts.html).

Once you have these things installed, then you can run the examples. Like this for instance:

```
python test1.py 0 640 480 5
```

The above runs ```test1.py``` on the first video device (i.e. /dev/video0) for 5 seconds.
