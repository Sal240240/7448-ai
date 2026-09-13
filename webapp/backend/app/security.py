"""Security middleware: response headers, request size limits, rate limiting.

Scoped to what this service actually is -- a stateless, read-only, no-auth API
in front of a fixed model. There are no sessions, no cookies and no database,
so the classic injection/auth surface largely doesn't exist here; what remains
worth defending is (a) the browser's treatment of responses, (b) resource
exhaustion, and (c) other origins driving this API from a visitor's browser.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# This API serves JSON only and is never framed, so the policy can be maximally
# restrictive: no scripts, no embedding, no plugins. The frontend is served
# separately and carries its own policy.
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    # cross-origin, not same-site: CORP is enforced separately from CORS, so
    # "same-site" would block a frontend hosted on a different registrable
    # domain even with a correct Access-Control-Allow-Origin — and it surfaces
    # as an opaque blocked response rather than a legible CORS error. Access is
    # governed by the explicit CORS allowlist; this data is public and
    # unauthenticated, so CORP is not the control doing the work.
    "Cross-Origin-Resource-Policy": "cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=(), interest-cohort=()",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, is_production: bool = False):
        super().__init__(app)
        self.is_production = is_production

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if self.is_production:
            # Only meaningful over HTTPS, and actively unhelpful on a local
            # http:// dev server, where it would pin the browser to HTTPS for
            # localhost across every project.
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


class RequestSizeLimitMiddleware:
    """Reject oversized bodies, including ones that declare no size.

    Checking Content-Length alone is not enough: a request sent with
    `Transfer-Encoding: chunked` carries no Content-Length, skips the header
    check entirely, and is then buffered into memory in full by the time
    Pydantic sees it. So a body arriving without a declared length is counted
    as it streams and refused the moment it crosses the limit.

    Written as raw ASGI rather than BaseHTTPMiddleware deliberately: reading
    the stream consumes it, and BaseHTTPMiddleware's `call_next` does not carry
    a substituted receive channel through to the endpoint, so a buffered body
    cannot be replayed downstream. At the ASGI layer the receive callable is
    ours to replace, which is the only place this can be done correctly.
    """

    BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in self.BODY_METHODS:
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    return await self._reject(send, 413, "Request body too large")
            except ValueError:
                return await self._reject(send, 400, "Invalid Content-Length")
            # The server enforces a declared Content-Length itself, so the body
            # can stream straight through without buffering it here.
            return await self.app(scope, receive, send)

        chunks, total, more_body = [], 0, True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_bytes:
                return await self._reject(send, 413, "Request body too large")
            chunks.append(chunk)
            more_body = message.get("more_body", False)

        body = b"".join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    @staticmethod
    async def _reject(send, status: int, detail: str) -> None:
        payload = json.dumps({"detail": detail}).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": payload})


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-client rate limit, held in process memory.

    Deliberately not Redis-backed: this service is a single stateless process
    whose whole point is that it stores nothing. If it is ever scaled to
    multiple replicas this becomes per-replica, which is a documented and
    acceptable weakening for a public read-only endpoint -- not a silent one.
    """

    max_tracked_clients = 10_000

    def __init__(self, app, max_requests: int, window_s: int, trust_proxy: bool = False):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_s = window_s
        self.trust_proxy = trust_proxy
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self._last_evict = 0.0

    def _client_key(self, request: Request) -> str:
        """Identify the caller, honoring X-Forwarded-For only behind a trusted proxy.

        Deployed behind nginx, request.client.host is the proxy for every
        request, which would put all visitors in a single shared bucket and
        effectively rate-limit the whole site as one client. The header fixes
        that -- but it is caller-controlled, so trusting it when NOT behind a
        proxy would let anyone bypass the limit by inventing an address. Hence
        opt-in via TRUST_PROXY_HEADERS, off by default.
        """
        if self.trust_proxy:
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded:
                # Leftmost entry is the original client; the rest are hops.
                return forwarded.split(",")[0].strip()[:64]
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        key = self._client_key(request)
        now = time.monotonic()
        window = self.hits[key]
        while window and now - window[0] > self.window_s:
            window.popleft()

        if len(window) >= self.max_requests:
            retry_after = max(1, int(self.window_s - (now - window[0])))
            return JSONResponse(
                {"detail": "Rate limit exceeded. Please slow down."},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)
        self._evict(now)

        return await call_next(request)

    def _evict(self, now: float) -> None:
        """Bound memory without turning into a per-request full scan.

        The obvious version -- scan every key whenever the dict is large and
        drop the expired ones -- degrades badly under a flood of never-repeated
        source addresses (trivial over IPv6). Once the dict is full of *active*
        keys, the scan finds nothing to remove and then runs again on the very
        next request, so cost grows with attacker-driven size while memory keeps
        growing anyway: strictly worse than doing nothing.

        Instead, sweep at most once per window, and if a sweep cannot get the
        dict under the cap, drop the oldest entries outright. Evicting a live
        attacker's counter only resets their budget; letting the process die
        denies everyone.
        """
        if len(self.hits) <= self.max_tracked_clients:
            return
        if now - self._last_evict < self.window_s and len(self.hits) < self.max_tracked_clients * 2:
            return
        self._last_evict = now

        for key in [k for k, v in self.hits.items() if not v or now - v[-1] > self.window_s]:
            del self.hits[key]

        if len(self.hits) > self.max_tracked_clients:
            oldest = sorted(self.hits, key=lambda k: self.hits[k][-1] if self.hits[k] else 0.0)
            for key in oldest[: len(self.hits) - self.max_tracked_clients]:
                del self.hits[key]
