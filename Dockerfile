# Python 3.8 is required: the pinned deps (numpy 1.19, scikit-learn 0.24 for the
# pickled models, opencv 4.5.1) only ship wheels up to Python 3.9.
FROM python:3.8-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libgl1 + libglib2.0-0: required by opencv-python at import time
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python manage.py collectstatic --noinput

EXPOSE 8000
# KMeans on a 600x400 image can take a few seconds -> generous timeout
CMD ["gunicorn", "colorful_classification_adaboost.wsgi:application", \
     "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "--access-logfile", "-"]
