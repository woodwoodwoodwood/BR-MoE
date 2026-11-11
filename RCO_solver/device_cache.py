"""Byte-bounded LRU for packed GPU bases and low-rank factors during search."""
from collections import OrderedDict


class DeviceCache:
    def __init__(self, max_bytes):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes = 0
        self.entries = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.entries.move_to_end(key)
        self.hits += 1
        return entry[0]

    def put(self, key, value):
        size = sum(t.numel() * t.element_size() for t in value)
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.bytes -= previous[1]
        if size > self.max_bytes:
            return
        while self.bytes + size > self.max_bytes:
            _, (_, removed_size) = self.entries.popitem(last=False)
            self.bytes -= removed_size
        self.entries[key] = (value, size)
        self.bytes += size
