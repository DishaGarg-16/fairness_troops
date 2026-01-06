import redis.asyncio as redis
import os
import json
import hashlib

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

async def get_redis_client():
    return redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

def generate_cache_key(model_bytes, data_bytes, config):
    """
    Generates a unique hash for the audit request inputs.
    """
    hasher = hashlib.sha256()
    hasher.update(model_bytes)
    hasher.update(data_bytes)
    # Sort keys to ensure consistent order
    config_str = json.dumps(config, sort_keys=True)
    hasher.update(config_str.encode('utf-8'))
    return f"audit:{hasher.hexdigest()}"
