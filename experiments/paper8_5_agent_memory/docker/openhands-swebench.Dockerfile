ARG BASE_IMAGE
FROM ghcr.io/astral-sh/uv:0.8.13 AS uv
FROM ${BASE_IMAGE}
COPY --from=uv /uv /uvx /usr/local/bin/
RUN apt-get update \
    && apt-get install -y --no-install-recommends tmux \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.12 \
    && uv venv /opt/openhands --python 3.12 \
    && uv pip install --python /opt/openhands/bin/python \
       openhands-sdk==1.49.2 openhands-tools==1.49.2 \
    && uv pip check --python /opt/openhands/bin/python
COPY experiments/paper8_5_agent_memory/openhands_swebench_entry.py \
     /opt/paper85/openhands_swebench_entry.py
COPY src/pra_hf/execution_receipts.py \
     /opt/paper85/execution_receipts.py
WORKDIR /testbed
