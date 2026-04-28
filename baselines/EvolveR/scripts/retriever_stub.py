#!/usr/bin/env python3
"""Minimal /retrieve that returns empty passage lists (no Wiki)."""
import os

from fastapi import FastAPI
from uvicorn import run

app = FastAPI()


@app.post("/retrieve")
def retrieve(payload: dict):
    queries = payload.get("queries") or []
    return {"result": [[] for _ in queries]}


if __name__ == "__main__":
    port = int(os.environ.get("RETRIEVER_STUB_PORT", "19999"))
    run(app, host="0.0.0.0", port=port, log_level="warning")
