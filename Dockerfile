FROM python:3.10-slim

RUN apt update && apt upgrade -y && apt install -y git curl

# Set working directory
WORKDIR /app

# Copy & install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for models
RUN mkdir -p models bow

# Copy service code
COPY . .

EXPOSE 5001

CMD ["python", "app/app.py"]