"""Read-only OpenAI-compatible client for the teacher model (DeepSeek V4.1 Flash served on this host).

Rules (docs/distill_design.md): the teacher service is never started, stopped, reconfigured or restarted from here;
only /v1/models (GET) and /v1/chat/completions (POST) are used, with bounded concurrency and a minimum interval
between requests so production users of the same server are not starved.
"""
import json, os, threading, time, urllib.request, urllib.error


class Teacher:
    def __init__(self, base_url, model, concurrency=2, min_interval=0.5, timeout=600, api_key=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sem = threading.Semaphore(concurrency)
        self.min_interval = min_interval
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("TEACHER_API_KEY")
        self._lock = threading.Lock()
        self._last = 0.0
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.errors = 0

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def probe(self):
        """GET /v1/models; returns the model list (read-only)."""
        req = urllib.request.Request(self.base_url + "/v1/models", headers=self._headers())
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
        ids = [m["id"] for m in d.get("data", [])]
        if self.model not in ids:
            raise RuntimeError(f"teacher model {self.model!r} not served; served: {ids}")
        return ids

    def chat(self, messages, max_tokens=2048, temperature=0.6, top_p=0.95, seed=None, thinking=False,
             response_format=None, retries=3):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": top_p}
        if seed is not None:
            body["seed"] = seed
        if thinking:
            body["thinking"] = True
        if response_format:
            body["response_format"] = response_format
        data = json.dumps(body).encode()
        for attempt in range(retries):
            with self.sem:
                with self._lock:
                    wait = self._last + self.min_interval - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    self._last = time.time()
                t0 = time.time()
                try:
                    req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=data, headers=self._headers())
                    with urllib.request.urlopen(req, timeout=self.timeout) as r:
                        out = json.loads(r.read())
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as e:
                    self.errors += 1
                    if attempt == retries - 1:
                        raise
                    time.sleep(5 * (attempt + 1))
                    continue
            ch = out["choices"][0]
            u = out.get("usage", {})
            with self._lock:
                self.calls += 1
                self.prompt_tokens += u.get("prompt_tokens", 0)
                self.completion_tokens += u.get("completion_tokens", 0)
            msg = ch.get("message", {})
            return {"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content"),
                    "finish_reason": ch.get("finish_reason"), "usage": u, "latency_s": round(time.time() - t0, 2),
                    "model": out.get("model")}

    def stats(self):
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "errors": self.errors}
