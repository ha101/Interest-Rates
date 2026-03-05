import pytest

from interest_rate_meta_model.cache import HTTPCache, HTTPCacheConfig


def test_clear_rejects_path_traversal_namespaces(tmp_path):
    cache = HTTPCache(HTTPCacheConfig(base_dir=tmp_path))
    for namespace in ("../x", "..", "/tmp", "foo/bar", "foo\\bar", ""):
        with pytest.raises(ValueError):
            cache.clear(namespace)


def test_clear_only_removes_requested_namespace(tmp_path):
    cache = HTTPCache(HTTPCacheConfig(base_dir=tmp_path))
    cache.set("treasury", "https://example.com/a", content=b"a")
    cache.set("fred", "https://example.com/b", content=b"b")

    cache.clear("treasury")

    treasury_dir = tmp_path / "treasury"
    fred_dir = tmp_path / "fred"
    assert (not treasury_dir.exists()) or (not any(treasury_dir.iterdir()))
    assert fred_dir.exists()
    assert any(fred_dir.iterdir())
