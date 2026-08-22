import pytest

from common.storage.local import LocalStorage
from common.storage.base import StorageBackend
from common.storage.gcs import GCSStorage
from common.storage.s3 import S3Storage
from common.storage.transfer import get_tree, put_tree, sha256_file


def test_local_storage_roundtrip_checksum_and_windows_key(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.txt").write_text("distributed PRA", encoding="utf-8")
    backend = LocalStorage("test", tmp_path / "storage")
    put_tree(backend, source, "run\\trial")
    destination = tmp_path / "download"
    get_tree(backend, "run/trial", destination)
    assert (destination / "value.txt").read_text(encoding="utf-8") == "distributed PRA"
    assert sha256_file(source / "value.txt") == sha256_file(destination / "value.txt")


def test_local_storage_rejects_path_escape(tmp_path):
    backend = LocalStorage("test", tmp_path / "storage")
    with pytest.raises(ValueError, match="escapes root"):
        backend.exists("../private")


def test_s3_backend_strips_configured_prefix_from_list(tmp_path):
    class Paginator:
        def paginate(self, **kwargs):
            assert kwargs["Prefix"] == "research/run"
            return [{"Contents": [{"Key": "research/run/metric.json"}]}]

    class Client:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return Paginator()

    backend = S3Storage.__new__(S3Storage)
    StorageBackend.__init__(backend, "mock")
    backend.client = Client()
    backend.bucket = "bucket"
    backend.prefix = "research"
    assert backend.list("run") == ["run/metric.json"]
    assert backend.uri("run") == "s3://bucket/research/run"


def test_gcs_backend_strips_configured_prefix_from_list():
    class Blob:
        name = "research/run/metric.json"

    class Client:
        def list_blobs(self, bucket, prefix):
            assert prefix == "research/run"
            return [Blob()]

    backend = GCSStorage.__new__(GCSStorage)
    StorageBackend.__init__(backend, "mock")
    backend.client = Client()
    backend.bucket = object()
    backend.bucket_name = "bucket"
    backend.prefix = "research"
    assert backend.list("run") == ["run/metric.json"]
    assert backend.uri("run") == "gs://bucket/research/run"
