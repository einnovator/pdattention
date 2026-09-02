FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[registry]"
EXPOSE 9200
CMD ["sh", "-c", "alembic -c alembic.ini upgrade head && pra-registry serve --config deploy/registry/registry.yaml"]
