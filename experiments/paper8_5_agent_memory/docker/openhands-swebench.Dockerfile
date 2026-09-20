ARG BASE_IMAGE
ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE} AS openhands_runtime
FROM ${BASE_IMAGE}
RUN apt-get update \
    && apt-get install -y --no-install-recommends tmux \
    && rm -rf /var/lib/apt/lists/*
COPY --from=openhands_runtime /root/.local/share/uv/python/ \
     /root/.local/share/uv/python/
COPY --from=openhands_runtime /opt/openhands/ /opt/openhands/
COPY experiments/paper8_5_agent_memory/openhands_swebench_entry.py \
     /opt/paper85/openhands_swebench_entry.py
COPY src/pra_hf/execution_receipts.py \
     /opt/paper85/execution_receipts.py
WORKDIR /testbed
