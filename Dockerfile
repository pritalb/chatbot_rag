FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y curl
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . /app

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]