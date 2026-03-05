from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


def _canonical_url(url: str, params: dict[str, Any] | None = None) -> str:
    if not params:
        return url
    items: list[tuple[str, str]] = []
    for key in sorted(params):
        value = params[key]
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for v in value:
                items.append((str(key), str(v)))
        else:
            items.append((str(key), str(value)))
    if not items:
        return url
    query = urlencode(items)
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{query}"


@dataclass(slots=True)
class HTTPCacheConfig:
    base_dir: str | Path = "~/.cache/interest-rate-meta-model"
    enabled: bool = True
    default_ttl_seconds: int | None = 12 * 60 * 60


class HTTPCache:
    """Very small file-backed cache for GET responses.

    The cache stores raw response content plus a metadata sidecar. Keys are derived
    from the fully qualified request URL including query parameters. Different data
    providers are separated into namespaces such as ``treasury`` or ``newyorkfed``.
    """

    def __init__(self, config: HTTPCacheConfig | None = None) -> None:
        self.config = config or HTTPCacheConfig()
        self.base_dir = Path(self.config.base_dir).expanduser().resolve()
        if self.config.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    _NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

    def _namespace_dir(self, namespace: str) -> Path:
        clean = str(namespace).strip()
        if not clean:
            raise ValueError("cache namespace must not be empty")
        if not self._NAMESPACE_RE.fullmatch(clean):
            raise ValueError("cache namespace contains unsupported characters")
        target = (self.base_dir / clean).resolve()
        if self.base_dir not in target.parents:
            raise ValueError("cache namespace must resolve under the cache directory")
        return target

    def _paths(self, namespace: str, canonical_url: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
        ns_dir = self._namespace_dir(namespace)
        data_path = ns_dir / f"{digest}.bin"
        meta_path = ns_dir / f"{digest}.json"
        return data_path, meta_path

    def get(
        self,
        namespace: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> tuple[bytes, dict[str, Any]] | None:
        if not self.config.enabled:
            return None
        ttl = self.config.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        canonical = _canonical_url(url, params)
        data_path, meta_path = self._paths(namespace, canonical)
        if not data_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        created_at = float(meta.get("created_at", 0.0))
        if ttl is not None and ttl >= 0 and (time.time() - created_at) > ttl:
            return None
        try:
            payload = data_path.read_bytes()
        except Exception:
            return None
        return payload, meta

    def set(
        self,
        namespace: str,
        url: str,
        *,
        content: bytes,
        params: dict[str, Any] | None = None,
        status_code: int = 200,
        headers: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        canonical = _canonical_url(url, params)
        data_path, meta_path = self._paths(namespace, canonical)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(content)
        meta: dict[str, Any] = {
            "url": canonical,
            "created_at": time.time(),
            "status_code": int(status_code),
            "headers": headers or {},
        }
        if extra_meta:
            meta.update(extra_meta)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    def get_text(
        self,
        namespace: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
        encoding: str = "utf-8",
    ) -> tuple[str, dict[str, Any]] | None:
        cached = self.get(namespace, url, params=params, ttl_seconds=ttl_seconds)
        if cached is None:
            return None
        content, meta = cached
        return content.decode(encoding, errors="replace"), meta

    def _remove_tree_contents(self, path: Path) -> None:
        for child in path.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink(missing_ok=True)
                continue
            if child.is_dir():
                self._remove_tree_contents(child)
                try:
                    child.rmdir()
                except OSError:
                    pass

    def clear(self, namespace: str | None = None) -> None:
        if not self.base_dir.exists():
            return
        target = self._namespace_dir(namespace) if namespace else self.base_dir
        if not target.exists():
            return
        self._remove_tree_contents(target)
        if namespace:
            try:
                target.rmdir()
            except OSError:
                pass
