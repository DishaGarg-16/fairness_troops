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
RUN uv pip install --system -r pyproject.toml

# Copy the rest of the application
COPY . .

# Install the project itself
RUN uv pip install --system -e .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
