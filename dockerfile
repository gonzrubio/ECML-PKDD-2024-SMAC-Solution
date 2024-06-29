# Start from a pytorch image
FROM pytorch/pytorch
WORKDIR /usr/local/app

RUN apt-get update
RUN apt-get install -y git

COPY ./solution .

RUN pip install -r requirements.txt

CMD [ "python", "docker_tester.py" ]
