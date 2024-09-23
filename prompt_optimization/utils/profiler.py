from time import time


class ProfilingHandler:
    def __init__(self, name: str):
        self.name = name
        self.events = []

    def add_event(self, name: str, start: float, duration: float):
        self.events.append({"name": name, "start": start, "duration": duration})
        self._persist()

    def _persist(self):
        with open(f"{self.name}.mermaid", "w") as f:
            f.write("gantt\n")
            for event in self.events:
                f.write(
                    f'    {event["name"]} :{event["start"]:.2f}, {event["duration"]:.2f}\n'
                )


class Profiler:
    def __init__(self, handler: ProfilingHandler, name: str):
        self.name = name
        self.handler = handler
        self.start = None
        self.duration = None

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.start

        self.duration = time() - self.start
        self.handler.add_event(self.name, self.start, self.duration)
