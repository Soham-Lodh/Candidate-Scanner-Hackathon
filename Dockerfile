FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500 \
    STREAMLIT_SERVER_MAX_MESSAGE_SIZE=500 \
    CANDIDATE_UPLOAD_HOST=0.0.0.0 \
    CANDIDATE_UPLOAD_PORT=8765 \
    CANDIDATE_UPLOAD_PUBLIC_URL=http://localhost:8765

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
EXPOSE 8765
CMD ["streamlit", "run", "app.py", "--server.maxUploadSize=500", "--server.maxMessageSize=500"]
