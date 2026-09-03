FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[router]"

EXPOSE 9400
ENTRYPOINT ["pra-router"]
CMD ["serve", "--config", "/etc/pra/reference-router.yaml", "--host", "0.0.0.0", "--port", "9400"]
