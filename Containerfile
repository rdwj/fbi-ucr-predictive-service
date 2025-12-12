# FBI UCR Crime Prediction Service
# Build: podman build --platform linux/amd64 -t fbi-ucr:latest -f Containerfile . --no-cache
# Run: podman run -p 8080:8080 fbi-ucr:latest

FROM registry.redhat.io/ubi9/python-311:latest

# Labels for OpenShift
LABEL name="fbi-ucr" \
      version="0.1.0" \
      summary="FBI UCR Crime Prediction Service" \
      description="Real-time crime prediction API using FBI Uniform Crime Reporting data" \
      io.k8s.display-name="FBI UCR Prediction Service" \
      io.openshift.tags="python,fastapi,ml,prediction"

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install dependencies
# Prophet requires cmdstan which needs specific setup
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "fastapi>=0.109.0" \
        "uvicorn[standard]>=0.27.0" \
        "pydantic>=2.5.0" \
        "pandas>=2.0.0" \
        "numpy>=1.24.0" \
        "statsmodels>=0.14.0" \
        "prophet>=1.1.5" \
        "joblib>=1.3.0" \
        "httpx>=0.26.0"

# Copy application code and fix permissions for OpenShift
COPY --chown=1001:0 src/ ./src/

# Copy trained models
COPY --chown=1001:0 models/ ./models/

# Ensure files are readable (OpenShift runs as random UID in root group)
RUN chmod -R g+rX ./src ./models

# Set environment variables
ENV PYTHONPATH=/app/src \
    MODELS_DIR=/app/models \
    HOST=0.0.0.0 \
    PORT=8080 \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "fbi_ucr.main:app", "--host", "0.0.0.0", "--port", "8080"]
