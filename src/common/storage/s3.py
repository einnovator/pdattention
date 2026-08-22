"""Optional Amazon S3-compatible storage backend."""

from __future__ import annotations

from pathlib import Path

from .base import StorageBackend


class S3Storage(StorageBackend):
    def __init__(
        self,
        name: str,
        *,
        bucket: str,
        prefix: str = "",
        profile: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
    ):
        super().__init__(name)
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("S3 storage requires the optional 'boto3' package.") from exc
        session = boto3.Session(profile_name=profile, region_name=region)
        self.client = session.client("s3", endpoint_url=endpoint_url)
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return "/".join(part for part in (self.prefix, key.strip("/")) if part)

    def put(self, local_path: str | Path, key: str) -> str:
        self.client.upload_file(str(local_path), self.bucket, self._key(key))
        return self.uri(key)

    def get(self, key: str, local_path: str | Path) -> Path:
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._key(key), str(target))
        return target

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except self.client.exceptions.ClientError:
            return False

    def list(self, prefix: str = ""):
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            for item in page.get("Contents", ()):
                key = item["Key"]
                if self.prefix and key.startswith(self.prefix + "/"):
                    key = key[len(self.prefix) + 1 :]
                keys.append(key)
        return sorted(keys)

    def uri(self, key: str = "") -> str:
        return f"s3://{self.bucket}/{self._key(key)}".rstrip("/")
