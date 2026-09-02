"""HTTP clients for the balancing API.

Two backends with the same interface:

* `LocalClient`  - the Docker image (`longhousedev/localapi`). One synchronous
  call per evaluation, no queue, no API key, no rate limit. Use this for search.
* `HostedClient` - the competition server. Asynchronous: submit, poll, retrieve.
  Use this to confirm final candidates against the official instance.
"""
from __future__ import annotations

import queue
import time
from typing import Any, Dict, List, Optional, Sequence

import requests

from .spec import Params


class ApiError(RuntimeError):
    pass


class RunRejected(ApiError):
    """The server refused the parameters — do not retry."""


class BaseClient:
    name = "base"

    def score(self, game: str, params: Params, run_type: str = "fast") -> float:
        raise NotImplementedError


class LocalClient(BaseClient):
    name = "local"

    def __init__(self, base_url: str = "http://localhost:3000/api/",
                 timeout_ms: Optional[int] = None, http_timeout: float = 900.0,
                 retries: int = 3):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_ms = timeout_ms
        self.http_timeout = http_timeout
        self.retries = retries
        self._session = requests.Session()

    def score(self, game: str, params: Params, run_type: str = "fast") -> float:
        body: Dict[str, Any] = {"game": game, "params": params, "run_type": run_type}
        if self.timeout_ms:
            body["timeout"] = self.timeout_ms
        last = None
        for attempt in range(self.retries):
            try:
                r = self._session.post(self.base_url + "run_game", json=body,
                                       timeout=self.http_timeout)
                data = r.json()
            except Exception as exc:  # transport hiccup — retry
                last = exc
                time.sleep(2 ** attempt)
                continue
            if "score" in data:
                return float(data["score"])
            err = data.get("error", data)
            if r.status_code == 400:
                raise RunRejected("local API rejected params: %s" % err)
            last = ApiError(str(err))
            time.sleep(2 ** attempt)
        raise ApiError("local API failed after %d attempts: %s" % (self.retries, last))


class LocalPoolClient(BaseClient):
    """Spread evaluations across several local API containers.

    Each container serialises its work — it runs one `java -jar TAG.jar` at a
    time — so a single instance gives no parallelism at all regardless of how
    many cores the host has. Throughput comes from running N containers on N
    ports and keeping exactly one request in flight per container, which is
    what this pool enforces.
    """

    name = "local-pool"

    def __init__(self, base_urls: Sequence[str], timeout_ms: Optional[int] = None,
                 http_timeout: float = 3600.0, retries: int = 3,
                 prune: bool = False, prune_timeout: float = 5.0):
        if not base_urls:
            raise ValueError("local pool needs at least one URL")
        urls = list(base_urls)
        if prune:
            urls = self._reachable(urls, prune_timeout)
            if not urls:
                raise ApiError("no pool endpoints reachable (checked %d)"
                               % len(base_urls))
        self.clients: List[LocalClient] = [
            LocalClient(u, timeout_ms, http_timeout, retries) for u in urls]
        self._free: "queue.Queue[LocalClient]" = queue.Queue()
        for c in self.clients:
            self._free.put(c)

    @staticmethod
    def _reachable(urls: Sequence[str], timeout: float) -> List[str]:
        """Keep only endpoints whose HTTP server answers at all. A remote box
        may still be booting containers; a dead port must not enter the pool
        and stall a worker for the full http_timeout."""
        import concurrent.futures as _cf
        sess = requests.Session()

        def ok(u: str) -> bool:
            try:  # any HTTP status (even 404/405) proves the server is up
                sess.get(u.rstrip("/"), timeout=timeout)
                return True
            except Exception:
                return False

        with _cf.ThreadPoolExecutor(max_workers=min(64, len(urls))) as ex:
            flags = list(ex.map(ok, urls))
        return [u for u, f in zip(urls, flags) if f]

    @property
    def size(self) -> int:
        return len(self.clients)

    def score(self, game: str, params: Params, run_type: str = "fast") -> float:
        client = self._free.get()
        try:
            return client.score(game, params, run_type)
        finally:
            self._free.put(client)


def pool_urls(size: int, first_port: int = 3000, host: str = "localhost") -> List[str]:
    return ["http://%s:%d/api/" % (host, first_port + i) for i in range(size)]


def parse_pool_spec(spec: str) -> List[str]:
    """Turn a compact pool spec into API URLs.

    Format: comma-separated ``host:first_port:count`` groups, e.g.
    ``"localhost:3000:10"`` or ``"1.2.3.4:3000:90,5.6.7.8:3000:90"``. Lets one
    search fan out across many containers on one or more cloud boxes.
    """
    urls: List[str] = []
    for group in spec.split(","):
        group = group.strip()
        if not group:
            continue
        host, first, count = group.rsplit(":", 2)
        urls.extend(pool_urls(int(count), int(first), host))
    return urls


class HostedClient(BaseClient):
    name = "hosted"

    # Statuses seen from the server while a run is still in flight. A `fast`
    # run takes ~3.5 minutes and sits in `created` until a worker picks it up,
    # so anything unrecognised is treated as "still pending" rather than an
    # error — aborting an unattended overnight search on an unknown status
    # string would be far more costly than waiting one extra poll.
    PENDING = ("created", "queued", "running", "results", "processing", "")

    def __init__(self, api_key: str,
                 base_url: str = "https://balance-competition.tabletopgames.ai/api/",
                 poll_interval: float = 10.0, max_wait: float = 14400.0,
                 http_timeout: float = 60.0, retries: int = 3):
        if not api_key:
            raise ValueError("hosted API requires an API key (see TTB_API_KEY)")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/"
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.http_timeout = http_timeout
        self.retries = retries
        self._session = requests.Session()

    def _request(self, method: str, path: str, **kw):
        """One HTTP call with retry on transport-level failures.

        The competition server drops connections under load ("connection reset
        by peer"), which surfaces as a raw requests exception. Left unwrapped it
        would escape as an uncaught traceback and kill a long unattended search,
        so every network error is retried with backoff and finally re-raised as
        an ApiError the search loop already knows how to absorb.
        """
        last = None
        for attempt in range(self.retries):
            try:
                return self._session.request(method, self.base_url + path,
                                             timeout=self.http_timeout, **kw)
            except requests.exceptions.RequestException as exc:
                last = exc
                time.sleep(2 ** attempt)
        raise ApiError("%s %s failed after %d attempts: %s"
                       % (method, path, self.retries, last))

    def submit(self, game: str, params: Params, run_type: str = "fast") -> int:
        body = {"game": game, "params": params, "api_key": self.api_key,
                "run_type": run_type}
        last = None
        for attempt in range(self.retries):
            r = self._request("POST", "submit_run", json=body,
                              headers={"Content-Type": "application/json"})
            if r.status_code == 200:
                return int(r.json()["runID"])
            if r.status_code == 401:
                raise ApiError("API key rejected (401)")
            if r.status_code == 400:
                raise RunRejected("server rejected params (400): %s" % r.text[:400])
            last = "%s %s" % (r.status_code, r.text[:200])
            time.sleep(2 ** attempt)
        raise ApiError("submit_run failed: %s" % last)

    def status(self, run_id: int) -> str:
        r = self._request("GET", "query_run", params={"id": run_id})
        if r.status_code != 200:
            raise ApiError("query_run %s: %s" % (r.status_code, r.text[:200]))
        data = r.json()
        return data.get("run_status", data.get("status", ""))

    def result(self, run_id: int) -> float:
        r = self._request("GET", "retrieve_result",
                          params={"id": run_id, "api_key": self.api_key})
        if r.status_code != 200:
            raise ApiError("retrieve_result %s: %s" % (r.status_code, r.text[:200]))
        return float(r.json()["score"])

    def score(self, game: str, params: Params, run_type: str = "fast") -> float:
        run_id = self.submit(game, params, run_type)
        deadline = time.time() + self.max_wait
        unknown = None
        while time.time() < deadline:
            time.sleep(self.poll_interval)
            try:
                st = self.status(run_id)
            except ApiError:
                continue  # transient; keep polling until max_wait
            if st == "complete":
                return self.result(run_id)
            if st in ("failed", "error", "cancelled"):
                raise ApiError("run %d reported status %r" % (run_id, st))
            if st not in self.PENDING and st != unknown:
                unknown = st
                print("  note: treating unknown run status %r as pending "
                      "(run %d)" % (st, run_id))
        raise ApiError("run %d did not complete within %.0fs" % (run_id, self.max_wait))


class ModalClient(BaseClient):
    """Evaluate via the deployed Modal function (modal_localapi.py).

    Unlike a web endpoint, a Modal function call runs up to the container
    timeout, so multi-minute evaluations complete in one call. Each call is one
    Modal input; with max_inputs=1 per container Modal fans concurrent calls out
    across the autoscaled fleet, so calling score() from N worker threads uses N
    containers.
    """

    name = "modal"

    def __init__(self, app_name: str = "ttb-localapi", cls_name: str = "Evaluator",
                 timeout_ms: Optional[int] = None, retries: int = 6):
        import modal  # lazy: only the modal backend needs the SDK
        self._modal = modal
        self._obj = modal.Cls.from_name(app_name, cls_name)()
        self.timeout_ms = timeout_ms or 0
        self.retries = retries

    def score(self, game: str, params: Params, run_type: str = "fast") -> float:
        last = None
        for attempt in range(self.retries):
            try:
                data = self._obj.run.remote(game, params, run_type, self.timeout_ms)
            except Exception as exc:  # transient Modal/infra/DNS error
                last = exc
                time.sleep(min(60, 3 * 2 ** attempt))   # up to ~1.5 min ridden out
                continue
            if isinstance(data, dict) and "score" in data:
                return float(data["score"])
            err = data.get("error", data) if isinstance(data, dict) else data
            raise RunRejected("modal eval rejected params: %s" % err)
        raise ApiError("modal call failed after %d attempts: %s"
                       % (self.retries, last))


def make_client(backend: str, api_key: str = "", local_url: str = "",
                hosted_url: str = "", pool: int = 1, first_port: int = 3000,
                **kw) -> BaseClient:
    if backend == "local":
        urls = kw.pop("pool_url_list", None)
        if urls:
            return LocalPoolClient(urls, prune=kw.pop("prune", True), **kw)
        if pool > 1:
            return LocalPoolClient(pool_urls(pool, first_port),
                                   prune=kw.pop("prune", False), **kw)
        kw.pop("prune", None)
        return LocalClient(local_url or "http://localhost:3000/api/", **kw)
    if backend == "hosted":
        return HostedClient(api_key, hosted_url or
                            "https://balance-competition.tabletopgames.ai/api/", **kw)
    if backend == "modal":
        return ModalClient(**kw)
    raise ValueError("unknown backend %r" % backend)
