FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY . /app

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]