# 1. Start from an official Python base image
FROM python:3.14-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements and install dependencies first
# This uses Docker's "layer caching" so it doesn't redownload packages every time you change your code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the application code into the container
COPY . .

# 5. Create necessary directories for state/data storage
RUN mkdir -p data training_set quarantine errors models

# 6. Command to start the orchestration loop
CMD ["python", "src/main.py"]