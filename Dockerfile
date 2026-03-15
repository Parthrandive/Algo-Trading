# Use a stable Python base with broad package compatibility.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

# Install dependencies and fail fast if critical market/data libs are missing.
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -c "import yfinance, nsepython, psycopg2, torch, statsmodels"

COPY . .

# Keep the container running for development purposes.
CMD ["tail", "-f", "/dev/null"]
