# Stage 1: Build environment
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files first to leverage Docker cache
COPY requirements.txt pyproject.toml setup.py ./
COPY README.md LICENSE ./

# Create virtual environment and activate it
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install the package in development mode
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime environment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/scripts ./scripts

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/data/{raw,processed,external} \
    /app/models/trained \
    /app/results/{predictions,plots,logs} \
    /app/cache \
    /app/notebooks \
    && chown -R nobody:nogroup /app \
    && chmod -R 755 /app

# Create volume mount points
VOLUME ["/app/data", "/app/models", "/app/results", "/app/cache", "/app/notebooks"]

# Switch to non-root user for security
USER nobody

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0 if all(hasattr(__import__('f1predictor'), attr) for attr in ['data', 'models', 'features']) else 1)"

# Default command
ENTRYPOINT ["python", "-m", "scripts.main_predictor"]
CMD ["--help"] 