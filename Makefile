VENV_LIB = venv/lib/python2.7

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate \
	&& pip install -r requirements.txt

CV2_SO = $(VENV_LIB)/cv2.so

LIB := $(shell python -c 'import cv2; print(cv2)' | awk '{print $$4}' | sed s:"['>]":"":g)

$(CV2_SO): $(LIB)
	ln -s $(LIB) $(VENV_LIB)/

playcv2: venv $(CV2_SO)
	. venv/bin/activate \
	&& python playcv2.py 0 640 480 5

clean:
	rm -rf venv
