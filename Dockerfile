FROM python:3.7

WORKDIR /code
COPY . /code
RUN pip install .
