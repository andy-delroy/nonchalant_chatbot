import os
import json
from redis.asyncio import Redis
from fastapi import Request
from typing import Dict

"""
Redis server set up for context memory using redis.asyncio
"""
REDIS = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=True)
CTX_TTL = 900  # 15 minutes key life-time

async def get_context(request: Request) -> Dict:
    sid = request.session.setdefault("sid", os.urandom(8).hex())
    key = f"ctx:{sid}"
    data = await REDIS.get(key)
    ctx = json.loads(data) if data else {}

    #shows sid being set and loaded ctx
    #ctx is a dictionary of filters
    print(f"[CTX] sid={sid}  loaded={ctx}")
    return ctx

async def save_context(request: Request, ctx: Dict):
    await REDIS.setex(f"ctx:{request.session['sid']}", CTX_TTL, json.dumps(ctx))
