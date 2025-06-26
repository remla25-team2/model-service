FROM python:3.10-slim AS builder

# no cache in image
RUN apt update && apt upgrade -y && apt install -y git curl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Copy & install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim AS production

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Set working directory
WORKDIR /app

# Create directories for models
RUN mkdir -p models bow

# Copy service code
COPY . .

EXPOSE 5001

CMD ["python", "app/app.py"]