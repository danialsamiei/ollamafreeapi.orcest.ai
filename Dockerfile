FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ollamafreeapi library (installed as editable package)
COPY ollamafreeapi/ ./ollamafreeapi/
COPY setup.py pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy the gateway application
COPY app/ ./app/

EXPOSE 10000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
