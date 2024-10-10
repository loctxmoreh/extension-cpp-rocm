build:
	python3 setup.py build_ext

install: 
	python3 setup.py develop

test: 
	python3 test/test_extension.py

clean: 
	pip uninstall extension-cpp -y
	rm -rf build/ dist/ extension-cpp.egg-info/

.PHONY: build install test	
