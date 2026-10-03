# TESS transit search pipeline
FROM python:3.13-slim

# No .pyc files, unbuffered logs (so progress shows up live),
# and a non-interactive matplotlib backend (no screen in a container)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install dependencies first: this layer is cached and only rebuilds
# when requirements.txt changes, not on every code edit
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

ENTRYPOINT ["python", "-m", "src.run"]
