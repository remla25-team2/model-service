FROM python:3.10-slim

# Accept version as build argument
ARG MODEL_VERSION=latest

RUN apt update && apt upgrade -y && apt install -y git curl

# Set working directory
WORKDIR /app

# Copy & install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download model artifacts from GitHub release
RUN mkdir -p models bow && \
    if [ "$MODEL_VERSION" = "latest" ]; then \
        DOWNLOAD_URL="https://github.com/remla25-team2/model-training/releases/latest/download"; \
    else \
        DOWNLOAD_URL="https://github.com/remla25-team2/model-training/releases/download/v${MODEL_VERSION}"; \
    fi && \
    curl -L "${DOWNLOAD_URL}/SentimentModel.pkl" -o models/SentimentModel.pkl && \
    curl -L "${DOWNLOAD_URL}/c1_BoW_Sentiment_Model.pkl" -o bow/c1_BoW_Sentiment_Model.pkl

# Copy service code
COPY . .

EXPOSE 5001

CMD ["python", "app.py"]