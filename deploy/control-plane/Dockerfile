FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[control-plane]"
COPY deploy/control-plane/control-plane.yaml /etc/pra/control-plane.yaml
EXPOSE 9300
CMD ["pra-control", "serve", "--config", "/etc/pra/control-plane.yaml"]
