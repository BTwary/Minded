# syntax=docker/dockerfile:1.6
FROM python:3.11.8-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r aaos \
    && useradd -r -g aaos -m -s /bin/bash aaos

# The certified dependency lock is the canonical zero-cost/full Python dependency
# set for local operation. Server requirements (psycopg2-binary, redis) are installed
# to support client-selected PostgreSQL and Redis production deployments.
COPY requirements-lock.txt /tmp/requirements-lock.txt
COPY requirements-server.txt /tmp/requirements-server.txt
RUN pip install --no-cache-dir -r /tmp/requirements-lock.txt -r /tmp/requirements-server.txt \
    && rm -f /tmp/requirements-lock.txt /tmp/requirements-server.txt


COPY . /app
COPY apps/api/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN mkdir -p /app/data /app/data_store /tmp/aaos \
    && chown -R aaos:aaos /app /tmp/aaos /usr/local/bin/docker-entrypoint.sh

USER aaos
EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "apps.api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
