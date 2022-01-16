FROM gcr.io/kaggle-gpu-images/python:latest

COPY pyproject.toml poetry.lock ./
RUN apt-get update && apt-get upgrade -y \
        && apt-get install -y \
        python-is-python3 \
        git

RUN pip install poetry

COPY ./ ./