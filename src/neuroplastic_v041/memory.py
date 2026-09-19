"""Finite replay used only by sleep, never by prediction or cold recall."""
from collections import deque


def example_bytes(example):
    return sum(t.numel()*t.element_size() for t in (example.cue, example.context, example.target)) + 32


class ReplayStore:
    def __init__(self, capacity=128, byte_capacity=32768):
        if capacity < 1 or byte_capacity < 1:
            raise ValueError('positive replay budgets required')
        self.records = deque()
        self.capacity, self.byte_capacity = capacity, byte_capacity
        self.bytes = self.peak_bytes = self.evicted = 0

    def append(self, example):
        record = example.clone()
        charge = example_bytes(record)
        if charge > self.byte_capacity:
            raise ValueError('record exceeds replay byte budget')
        while self.records and (len(self.records) >= self.capacity or self.bytes+charge > self.byte_capacity):
            self.bytes -= example_bytes(self.records.popleft())
            self.evicted += 1
        self.records.append(record)
        self.bytes += charge
        self.peak_bytes = max(self.peak_bytes, self.bytes)

    def snapshot(self):
        return tuple(record.clone() for record in self.records)

    def retrieve_answer(self, *args, **kwargs):
        raise RuntimeError('episodic answer retrieval is disabled in this profile')
