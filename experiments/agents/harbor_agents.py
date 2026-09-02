"""Small Harbor adapters needed for reproducible PRA benchmark execution.

The execution path, trajectory parser, and verifier remain Harbor's.  This
module only replaces OpenCode's default ``nvm`` bootstrap because that path
clones GitHub from inside every task container and is brittle on restricted
benchmark hosts.
"""

from __future__ import annotations

from harbor.agents.installed.opencode import OpenCode
from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseEnvironment


NODE_VERSION = "22.22.0"
NODE_SHA256 = {
    "arm64": "1bf1eb9ee63ffc4e5d324c0b9b62cf4a289f44332dfef9607cea1a0d9596ba6f",
    "x64": "9aa8e9d2298ab68c600bd6fb86a6c13bce11a4eca1ba9b39d79fa021755d7c37",
}


def _pinned_node_command(*, package: str, executable: str) -> str:
    """Build the root command shared by Node-based Harbor agent adapters."""

    return (
        "set -euo pipefail; "
        "if command -v apt-get >/dev/null 2>&1; then "
        "apt-get update && apt-get install -y ca-certificates xz-utils; "
        "elif command -v apk >/dev/null 2>&1; then "
        "apk add --no-cache ca-certificates xz; "
        "else echo 'No supported package manager for xz' >&2; exit 2; fi; "
        "case \"$(uname -m)\" in "
        f"x86_64|amd64) arch=x64; checksum={NODE_SHA256['x64']} ;; "
        f"aarch64|arm64) arch=arm64; checksum={NODE_SHA256['arm64']} ;; "
        "*) echo \"Unsupported Node architecture: $(uname -m)\" >&2; exit 2 ;; "
        "esac; "
        f"node_dir=/opt/node-v{NODE_VERSION}-linux-$arch; "
        f"archive=node-v{NODE_VERSION}-linux-$arch.tar.xz; "
        f"url=https://nodejs.org/dist/v{NODE_VERSION}/$archive; "
        "curl -fsSL \"$url\" -o /tmp/node.tar.xz; "
        "echo \"$checksum  /tmp/node.tar.xz\" | sha256sum -c -; "
        "rm -rf \"$node_dir\"; "
        "tar -xJf /tmp/node.tar.xz -C /opt; "
        "ln -sf \"$node_dir/bin/node\" /usr/local/bin/node; "
        "ln -sf \"$node_dir/bin/npm\" /usr/local/bin/npm; "
        "ln -sf \"$node_dir/bin/npx\" /usr/local/bin/npx; "
        f"\"$node_dir/bin/npm\" install -g {package}; "
        f"ln -sf \"$node_dir/bin/{executable}\" /usr/local/bin/{executable}; "
        f"node --version; {executable} --version"
    )


class PinnedNodeOpenCode(OpenCode):
    """Install OpenCode with a checksum-pinned Node binary, then run normally."""

    async def install(self, environment: BaseEnvironment) -> None:
        """Install Node without GitHub, preserving Harbor's OpenCode behavior."""

        await self.ensure_system_dependencies(
            environment, ("curl", "bash", "coreutils")
        )
        version_spec = f"@{self._version}" if self._version else "@latest"
        command = _pinned_node_command(
            package=f"opencode-ai{version_spec}", executable="opencode"
        )
        await self.exec_as_root(environment, command=command)


class PinnedNodePi(Pi):
    """Install Pi without `nvm`, preserving Harbor's Pi execution path."""

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(
            environment, ("curl", "bash", "coreutils")
        )
        version_spec = f"@{self._version}" if self._version else "@latest"
        command = _pinned_node_command(
            package=f"--ignore-scripts {self._package_name()}{version_spec}",
            executable="pi",
        )
        await self.exec_as_root(environment, command=command)
