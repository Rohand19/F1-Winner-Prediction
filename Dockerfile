# Stage 1: Build environment
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project for installation
COPY . .

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .[dev]

# Stage 2: Runtime environment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source code and scripts
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create necessary directories with proper permissions
RUN mkdir -p /app/data/{raw,processed,external} \
    /app/models/trained \
    /app/results/{predictions,plots,logs} \
    /app/cache \
    && chown -R nobody:nogroup /app

# Switch to non-root user
USER nobody

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

# Default command
ENTRYPOINT ["python", "-m", "scripts.main_predictor"]
CMD ["--help"] 