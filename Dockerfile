FROM python:3.10-slim
WORKDIR /app

# The lock rather than the manifest, and hashes required rather than optional.
# The manifest fixes nine distributions; the install is fifty-four, and the
# thirty-odd it does not name were resolved freshly on every build — under a
# package whose stated reason for pinning is that a scoring result must be
# reproducible. `--require-hashes` also refuses anything the index serves that
# differs by a byte from what was resolved, which a version pin does not notice.
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

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
