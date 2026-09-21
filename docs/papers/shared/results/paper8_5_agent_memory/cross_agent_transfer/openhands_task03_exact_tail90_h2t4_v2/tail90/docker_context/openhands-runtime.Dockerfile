FROM ghcr.io/astral-sh/uv:0.8.13 AS uv
FROM debian:bookworm-slim
COPY --from=uv /uv /uvx /usr/local/bin/
RUN uv python install 3.12 \
    && uv venv /opt/openhands --python 3.12 \
    && uv pip install --python /opt/openhands/bin/python \
       openhands-sdk==1.49.2 openhands-tools==1.49.2 \
    && uv pip check --python /opt/openhands/bin/python
