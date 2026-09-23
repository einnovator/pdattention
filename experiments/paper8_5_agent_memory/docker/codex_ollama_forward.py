"""Transparent TCP bridge from Codex's local Ollama port to a remote host."""

from __future__ import annotations

import asyncio
import os


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def _handle(
    local_reader: asyncio.StreamReader,
    local_writer: asyncio.StreamWriter,
) -> None:
    host = os.environ["PRA_OLLAMA_FORWARD_HOST"]
    port = int(os.environ["PRA_OLLAMA_FORWARD_PORT"])
    try:
        remote_reader, remote_writer = await asyncio.open_connection(host, port)
    except Exception:
        local_writer.close()
        await local_writer.wait_closed()
        return
    await asyncio.gather(
        _pipe(local_reader, remote_writer),
        _pipe(remote_reader, local_writer),
        return_exceptions=True,
    )


async def _main() -> None:
    server = await asyncio.start_server(_handle, "127.0.0.1", 11434)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(_main())

