FROM python:3.10.12-slim-bullseye

WORKDIR /api-docker

COPY ./avapix/web_api/* .
COPY ./requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]