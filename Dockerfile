FROM tensorflow/tensorflow:1.15.0-py3
WORKDIR /code

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "prodserver.py"]