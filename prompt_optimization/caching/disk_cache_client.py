from diskcache import Cache


class DiskCacheClient:
    def __init__(self, cache_name: str):
        # 10 gb size limit
        self.cache = Cache(cache_name, size_limit=10**10)

    async def get(self, key):
        return self.cache.get(key)

    async def set(self, key, value, ttl=86400 * 30):
        self.cache.set(key, value, expire=ttl)
