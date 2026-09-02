"""TAG evaluation as a Modal *function* (not a web endpoint).

Modal's synchronous web requests time out around 150 s, but a single evaluation
can take minutes (CantStop fast ~3.5 min, Dominion much longer). Function calls
run up to the container ``timeout`` instead, so we expose the evaluation as a
Modal method and call it over the SDK.

Each container starts the competition's node server once and forwards one
request at a time (``max_inputs=1``) to it — reproducing the local
one-java-per-container pool — autoscaled to ``N_CONTAINERS``.

Deploy:   modal deploy modal_localapi.py
Use:      TTB_BACKEND=modal python -m ttbalance search --game Wonders7 ...
"""
import os
import subprocess
import time
import urllib.error
import urllib.request

import modal

N_CONTAINERS = 50
STARTUP_TIMEOUT = 240
EVAL_TIMEOUT = 3600

app = modal.App("ttb-localapi")
image = (
    modal.Image.from_registry("longhousedev/localapi", add_python="3.11")
    .pip_install("requests")
)


@app.cls(
    image=image,
    cpu=1.0,
    memory=1536,
    max_containers=N_CONTAINERS,
    timeout=EVAL_TIMEOUT + 120,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=1)   # one evaluation per container
class Evaluator:
    @modal.enter()
    def start(self):
        # Send the server's stdout to a file we can tail per request: TAG prints
        # the score's components ("Matrix Distance", "FPA Diff") there, and those
        # two numbers say WHICH part of the target a config is missing — a
        # diagnostic the HTTP API's single score throws away.
        self._log = open("/tmp/tag.log", "w+b", buffering=0)
        self.proc = subprocess.Popen(["node", "/out-obfuscated.js"], cwd="/",
                                     stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            try:
                urllib.request.urlopen("http://localhost:3000/", timeout=2)
                return
            except urllib.error.HTTPError:
                return  # any HTTP response (even 404) => server up
            except Exception:
                time.sleep(1)

    @modal.method()
    def run(self, game: str, params: dict, run_type: str = "fast",
            timeout_ms: int = 0) -> dict:
        import re
        import requests
        body = {"game": game, "params": params, "run_type": run_type}
        if timeout_ms:
            body["timeout"] = timeout_ms
        start = os.path.getsize("/tmp/tag.log") if os.path.exists("/tmp/tag.log") else 0
        r = requests.post("http://localhost:3000/api/run_game", json=body,
                          timeout=EVAL_TIMEOUT)
        try:
            out = r.json()
        except Exception:
            return {"error": "non-JSON (%s): %s" % (r.status_code, r.text[:200])}
        # Return the raw tail too: TAG prints the full win-rate matrix as a box
        # table, and knowing WHICH cell misses its target is far more actionable
        # than the scalar distance.
        try:
            with open("/tmp/tag.log", "rb") as fh:
                fh.seek(start)
                tail = fh.read().decode("utf-8", "replace")
            out["raw_tail"] = tail[-4000:]
            for key, pat in (("matrix_distance", r"Matrix Distance:\s*([-\d.]+)"),
                             ("fpa_diff", r"FPA Diff:\s*([-\d.]+)"),
                             ("total_distance", r"Total Distance:\s*([-\d.]+)"),
                             ("scaled_distance", r"Scaled Distance:\s*([-\d.]+)")):
                m = re.findall(pat, tail)
                if m:
                    out[key] = float(m[-1])
        except Exception:
            pass
        return out
