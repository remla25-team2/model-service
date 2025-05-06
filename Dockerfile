FROM python:3.10-slim

RUN apt update && apt upgrade -y && apt install -y git


# Set working directory

# Copy & install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY . .

# Copy model and vectorizer artifacts
COPY models/ ./models/
COPY bow/ ./bow/

WORKDIR /app


EXPOSE 5001

CMD ["python", "app.py"]