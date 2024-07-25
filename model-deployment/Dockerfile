FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app.py .
COPY ../nn-model-training/stock_price_model.h5 .
COPY ../statistical-model-training/arima_model.pkl .

CMD ["python", "app.py"]
