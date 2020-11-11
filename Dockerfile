FROM python:3.7-alpine

WORKDIR /app

# only copy requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app

RUN pip --no-cache-dir install -r requirements.txt

COPY . /app

CMD ["python","run.py"]
