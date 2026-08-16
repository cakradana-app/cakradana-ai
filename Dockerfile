FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY cakradana/ cakradana/
RUN pip install --no-cache-dir --no-deps .

# Model artifacts are mounted rather than baked in. They are versioned
# separately from the code and a rebuild must not silently change which model
# is serving; the service runs on rules alone when none is mounted, and says so
# on its readiness endpoint.
VOLUME ["/app/artifacts"]

EXPOSE 8000
CMD ["uvicorn", "cakradana.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
