SHELL=/bin/bash
CUML_HOME=./cuml

POETRY = pip3 install -q poetry

CUML = git clone https://github.com/rapidsai/cuml.git ${CUML_HOME} \
		&& cd ${CUML_HOME}/cpp \
		&& mkdir -p build && cd build \
		&& export CUDA_BIN_PATH=${CUDA_HOME} \
		&& cmake .. \
		&& make install \
		&& cd ../../python \
		&& python setup.py build_ext --inplace \
		&& python setup.py install

PACAKGE = black mypy isort flake8 python-box bbox-utility norfair==0.3.1
TORCH = torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

poetry:
	${POETRY}

setup: ## setup package on kaggle docker image
	python -m pip install ${PACAKGE} \
	&& python -m pip install ${TORCH} \
	&& sh ./install-yolov5.sh

develop: # usually use this command
	${POETRY} \
	&& poetry install \
	&& poe force-cuda11


develop_no_venv:
	${POETRY} \
	&& poetry config virtualenvs.create false \
	&& poetry install \

set_tpu:
	${POETRY} \
	&& poetry config virtualenvs.create false --local \
	&& poetry install \
	&& poetry run python3 -m pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl \

cuml:
	${CUML}

pip_export:
	pip3 freeze > requirements.txt

poetry_export:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

develop_by_requirements:
	for package in $(cat requirements.txt); do poetry add "${package}"; done

download_datasets: ## download datasets (you need kaggle-api)
	python ./src/tasks/fetch_datasets.py --compe_name tensorflow-great-barrier-reef

update_datasets:
	zip -r output/sub.zip output/sub
	kaggle datasets version -p ./output/sub -m "Updated data"

pull_kaggle_image:
	docker pull gcr.io/kaggle-gpu-images/python

build_dev_image:
	docker build -f Dockerfile -t local-kaggle-python .
