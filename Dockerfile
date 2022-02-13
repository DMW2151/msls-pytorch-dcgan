FROM python:3.8-slim-buster

# Copy model training code to container
COPY ./model/ ./model

RUN python3 -m pip install --upgrade pip &&\
    pip3 install /model \
        flask \
        flask-cors

# Copy model API code to container
COPY ./api ./api/

# Run API...
RUN python3 ./api/img_svc.py
