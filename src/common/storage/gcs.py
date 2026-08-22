"""Optional Google Cloud Storage artifact backend."""

from __future__ import annotations

from pathlib import Path

from .base import StorageBackend


class GCSStorage(StorageBackend):
    def __init__(
        self,
        name: str,
        *,
        bucket: str,
        prefix: str = "",
        project: str | None = None,
        credentials_file: str | None = None,
    ):
        super().__init__(name)
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise RuntimeError(
                "GCS storage requires the optional 'google-cloud-storage' package."
            ) from exc
        if credentials_file:
            self.client = storage.Client.from_service_account_json(
                str(Path(credentials_file).expanduser()), project=project
            )
        else:
            self.client = storage.Client(project=project)
        self.bucket = self.client.bucket(bucket)
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return "/".join(part for part in (self.prefix, key.strip("/")) if part)

    def put(self, local_path: str | Path, key: str) -> str:
        self.bucket.blob(self._key(key)).upload_from_filename(str(local_path))
        return self.uri(key)

    def get(self, key: str, local_path: str | Path) -> Path:
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.bucket.blob(self._key(key)).download_to_filename(str(target))
        return target

    def exists(self, key: str) -> bool:
        return bool(self.bucket.blob(self._key(key)).exists())

    def list(self, prefix: str = ""):
        keys = []
        for blob in self.client.list_blobs(self.bucket, prefix=self._key(prefix)):
            key = blob.name
            if self.prefix and key.startswith(self.prefix + "/"):
                key = key[len(self.prefix) + 1 :]
            keys.append(key)
        return sorted(keys)

    def uri(self, key: str = "") -> str:
        return f"gs://{self.bucket_name}/{self._key(key)}".rstrip("/")
