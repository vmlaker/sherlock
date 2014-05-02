########################################################
#
#  Makefile to install and run Sherlock.
#
########################################################

# Some of these may need adjusting on your specific system.
VENV_LIBDIR = venv/lib/python2.7
VENV_OPENCV = $(VENV_LIBDIR)/cv2.so
VENV_SHAREDMEM = $(VENV_LIBDIR)/site-packages/sharedmem

# This makefile uses bash shell commands.
SHELL := /bin/bash

# GLOBAL_OPENCV is the path to the system-installed OpenCV
# library cv2.so file. By default, the path is functionally
# obtained by printing the imported cv2 module in a Python shell.
#
# Alternatively, set the path explicitly. For OpenCV installation
# via Homebrew on OS X for example:
#
# GLOBAL_OPENCV := /usr/local/lib/python2.7/site-packages/cv2.so
GLOBAL_OPENCV := $(shell python -c 'import cv2; print(cv2)' | awk '{print $$4}' | sed s:"['>]":"":g)

all: venv $(VENV_SHAREDMEM) $(VENV_OPENCV)

# Link the global cv2 library file inside the virtual environment.
$(VENV_OPENCV): $(GLOBAL_OPENCV) venv
	cp -f $(GLOBAL_OPENCV) $@

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -r requirements.txt

# Install NumPy sharedmem module inside the virtual environment.
$(VENV_SHAREDMEM): venv
	. venv/bin/activate && \
	mkdir -p temp && \
	cd temp && \
	hg clone http://bitbucket.org/cleemesser/numpy-sharedmem && \
	cd numpy-sharedmem && \
	python setup.py install && \
	cd ../.. && \
	rm -rf temp

playcv2: venv $(VENV_OPENCV)
	. venv/bin/activate && python src/playcv2.py 0 640 480 8

diffavg1: venv $(VENV_OPENCV)
	. venv/bin/activate && python src/diffavg1.py 0 640 480 8

diffavg2: venv $(VENV_OPENCV)
	. venv/bin/activate && python src/diffavg2.py 0 640 480 8

diffavg3: venv $(VENV_SHAREDMEM) $(VENV_OPENCV)
	. venv/bin/activate && python src/diffavg3.py 0 640 480 8

diffavg4: venv $(VENV_SHAREDMEM) $(VENV_OPENCV)
	. venv/bin/activate && python src/diffavg4.py 0 640 480 8

object1: venv $(VENV_OPENCV)
	. venv/bin/activate && python src/object1.py 0 640 480 8

object2: venv $(VENV_SHAREDMEM) $(VENV_OPENCV)
	. venv/bin/activate && python src/object2.py 0 640 480 8

clean:
	rm -rf venv temp
