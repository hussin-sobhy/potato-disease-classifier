FROM python:3.10-slim

# Install system dependencies (OpenCV, etc.)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app


COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
# Copy all your project files (api, models, etc.) into /app
COPY . /app

# Switch working directory to 'api' inside the container
WORKDIR /app/api

# Expose port 8080
EXPOSE 8080

# Start FastAPI on port 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
