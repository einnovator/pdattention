ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG NODE_VERSION=22.20.0
ARG CODEX_VERSION=0.153.0
RUN curl --fail --silent --show-error --location \
      "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
      --output /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz --directory /usr/local --strip-components=1 \
    && rm /tmp/node.tar.xz
RUN npm install --global --no-audit --no-fund "@openai/codex@${CODEX_VERSION}" \
    && codex --version
WORKDIR /testbed

