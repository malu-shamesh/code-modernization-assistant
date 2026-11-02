"""``python -m modernizer.service``: serve on ``$PORT`` (default 8080)."""

from __future__ import annotations

import os

import uvicorn

DEFAULT_PORT = 8080


def main() -> None:
    raw = os.environ.get("PORT", "").strip()
    try:
        port = int(raw) if raw else DEFAULT_PORT
    except ValueError:
        raise SystemExit(f"PORT must be an integer, got {raw!r}") from None
    uvicorn.run("modernizer.service.app:create_app", factory=True, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
