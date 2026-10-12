from collections import OrderedDict


class BoundedCache:
    def __init__(self, max_entries: int, max_bytes: int):
        if any(type(value) is not int or value < 0 for value in (max_entries, max_bytes)):
            raise ValueError("cache entry and byte limits must be nonnegative integers")
        self.max_entries, self.max_bytes = int(max_entries), int(max_bytes)
        self.entries = OrderedDict()
        self.nbytes = self.hits = self.misses = 0

    def get(self, key):
        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return self.entries[key]

    def put(self, key, value):
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.nbytes -= previous.nbytes
        if not self.max_entries or not self.max_bytes or value.nbytes > self.max_bytes:
            return
        self.entries[key] = value
        self.nbytes += value.nbytes
        while len(self.entries) > self.max_entries or self.nbytes > self.max_bytes:
            _, evicted = self.entries.popitem(last=False)
            self.nbytes -= evicted.nbytes

    def clear(self):
        self.entries.clear()
        self.nbytes = self.hits = self.misses = 0

    def info(self):
        return {"entries": len(self.entries), "bytes": self.nbytes,
                "hits": self.hits, "misses": self.misses,
                "max_entries": self.max_entries, "max_bytes": self.max_bytes}
