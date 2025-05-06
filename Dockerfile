FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy & install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY app/ .

# Copy model and vectorizer artifacts
COPY models/ ./models/
COPY bow/ ./bow/


EXPOSE 5001

CMD ["python", "app.py"]