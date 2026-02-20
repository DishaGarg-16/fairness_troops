FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy requirements first to leverage caching
# Copy project definition
COPY pyproject.toml .

# Install dependencies using uv
# --system: install into system python
# --all-extras: install optional dependencies if any (we have celery[redis] in basics though)


# Copy the rest of the application
COPY . .

# Install the project itself (Standard non-editable install for production)
RUN uv pip install --system .

# Make this the default CMD, but overrideable by docker-compose
CMD ["python", "api/run_server.py"]
