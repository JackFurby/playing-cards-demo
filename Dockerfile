# syntax=docker/dockerfile:1

FROM python:3.10-slim-bullseye

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 libxrender-dev -y

WORKDIR /demo

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /demo

CMD [ "flask", "--app", "playing_cards", "run", "--host", "0.0.0.0"]
