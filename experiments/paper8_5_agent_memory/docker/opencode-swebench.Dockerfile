ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG NODE_VERSION=22.20.0
ARG OPENCODE_VERSION=1.18.31
RUN curl --fail --silent --show-error --location \
      "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
      --output /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz --directory /usr/local --strip-components=1 \
    && rm /tmp/node.tar.xz \
    && node --version \
    && npm --version
RUN npm install --global --no-audit --no-fund \
    "opencode-ai@${OPENCODE_VERSION}" \
    && opencode --version
WORKDIR /testbed
