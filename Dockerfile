# Build stage
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Final stage
FROM python:3.10-slim
WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /root/.local /root/.local
# Copy application files
COPY app.py version.txt ./
# Ensure pip scripts are in PATH
ENV PATH=/root/.local/bin:$PATH
# Expose configurable port
EXPOSE 5001
# Run the application
CMD ["python", "app.py"]