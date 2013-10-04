# Makefile for Sherlock.

# Some of these may need adjusting on specific systems.
VENV_LIB = venv/lib/python2.7
CV2_SO = $(VENV_LIB)/cv2.so
SHAREDMEM = $(VENV_LIB)/site-packages/sharedmem

all: venv $(SHAREDMEM) $(CV2_SO)

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -r requirements.txt

# Find OpenCV's cv2 library file for the global Python installation.
CV2_LIB := $(shell python -c 'import cv2; print(cv2)' | awk '{print $$4}' | sed s:"['>]":"":g)

# Link the global cv2 library file inside the virtual environment.
$(CV2_SO): $(CV2_LIB)
	ln -s $(CV2_LIB) $(VENV_LIB)/

# Install NumPy sharedmem module inside the virtual environment.
$(SHAREDMEM): venv
	. venv/bin/activate && \
	mkdir temp && pushd temp && \
	hg clone http://bitbucket.org/cleemesser/numpy-sharedmem && \
	cd numpy-sharedmem && \
	python setup.py install && \
	popd && rm -rf temp

playcv2: venv $(CV2_SO)
	. venv/bin/activate && python src/playcv2.py 0 640 480 8

diffavg1: venv $(CV2_SO)
	. venv/bin/activate && python src/diffavg1.py 0 640 480 8

diffavg2: venv $(CV2_SO)
	. venv/bin/activate && python src/diffavg2.py 0 640 480 8

diffavg3: venv $(SHAREDMEM) $(CV2_SO)
	. venv/bin/activate && python src/diffavg3.py 0 640 480 8

diffavg4: venv $(SHAREDMEM) $(CV2_SO)
	. venv/bin/activate && python src/diffavg4.py 0 640 480 8

object1: venv $(CV2_SO)
	. venv/bin/activate && python src/object1.py 0 640 480 8

object2: venv $(CV2_SO)
	. venv/bin/activate && python src/object2.py 0 640 480 8

clean:
	rm -rf venv
