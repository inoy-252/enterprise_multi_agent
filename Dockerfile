# Use official lightweight Python 3.11 slim image
FROM python:3.11-slim
# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set the working directory inside the container
WORKDIR /app
# Install dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy application source code
COPY multi_agent_supervisor.py .
COPY semantic_cache.py .
COPY api.py .
# Expose FastAPI default port
EXPOSE 8000
# Run Uvicorn production server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]