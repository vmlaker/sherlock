VENV_LIB = venv/lib/python2.7

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install -r requirements.txt

CV2_SO = $(VENV_LIB)/cv2.so
SHAREDMEM = $(VENV_LIB)/site-packages/sharedmem

LIB := $(shell python -c 'import cv2; print(cv2)' | awk '{print $$4}' | sed s:"['>]":"":g)

$(CV2_SO): $(LIB)
	ln -s $(LIB) $(VENV_LIB)/

$(SHAREDMEM): venv
	. venv/bin/activate && \
	mkdir temp && pushd temp && \
	hg clone http://bitbucket.org/cleemesser/numpy-sharedmem && \
	cd numpy-sharedmem && \
	python setup.py install && \
	popd && rm -rf temp

playcv2: venv $(CV2_SO)
	. venv/bin/activate && python playcv2.py 0 640 480 8

diffavg1: venv $(CV2_SO)
	. venv/bin/activate && python diffavg1.py 0 640 480 8

diffavg2: venv $(CV2_SO)
	. venv/bin/activate && python diffavg2.py 0 640 480 8

diffavg3: venv $(SHAREDMEM) $(CV2_SO)
	. venv/bin/activate && python diffavg3.py 0 640 480 8

diffavg4: venv $(SHAREDMEM) $(CV2_SO)
	. venv/bin/activate && python diffavg4.py 0 640 480 8

object1: venv $(CV2_SO)
	. venv/bin/activate && python object1.py 0 640 480 8

object2: venv $(CV2_SO)
	. venv/bin/activate && python object2.py 0 640 480 8

clean:
	rm -rf venv
