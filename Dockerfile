FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy model (must be built before docker build)
COPY models/spam_classifier/ ./models/spam_classifier/

ENTRYPOINT ["python", "-m", "src.predict"]
