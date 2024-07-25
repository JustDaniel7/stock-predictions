FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY nn_model_training.py .
COPY ../data-pipeline/preprocessed_data.csv .

CMD ["python", "nn_model_training.py"]
