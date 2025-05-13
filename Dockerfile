FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml pyproject.toml
COPY pdm.lock pdm.lock

RUN pip install --no-cache-dir pdm
RUN pdm install --prod

COPY src src

CMD ["pdm", "run", "prod"]