import threading
from typing import Callable

# Registered providers.
providers = {}

# Instantiated tasks.
tasks = {}
taskLock = threading.Lock()


class StatusHolder:
    def __init__(self):
        self.status = ""
        self.is_done = threading.Event()


class TaskProvider:
    def Run(self, status_holder: StatusHolder):
        pass


class _Task(StatusHolder):
    def __init__(self, name: str, provider: TaskProvider):
        super().__init__()
        self.name = name
        self.provider = provider
        self.status = "New"
        self.result = None
        self.thread = None

    def Start(self):
        if self.thread:
            raise RuntimeError("Don't call Start twice!")
        self.thread = threading.Thread(target=self._Run, daemon=True)
        self.thread.start()

    def Stop(self, join=False):
        if not self.thread:
            raise RuntimeError("Don't call Stop before Start!")
        if self.is_done.is_set():
            return
        self.is_done.set()
        if join:
            self.thread.join()

    def _Run(self):
        self.status = "Started"
        self.result = self.provider.Run(self)
        self.status = "Done (" + self.status + ")"
        self.is_done.set()


def GetTask(name: str):
    with taskLock:
        if name in tasks:
            return tasks.get(name)
        return None


def GetOrCreateTask(name: str, provider: TaskProvider):
    with taskLock:
        if name in tasks and not tasks.get(name).is_done.is_set():
            return tasks.get(name)
        t = _Task(name, provider)
        tasks[name] = t
        return t


def GetProvider(name: str):
    if name not in providers:
        return None
    return providers[name]


class _WaitTask(TaskProvider):
    def __init__(self, json):
        self.waitTime = json["time"] or 10.0

    def Run(self, status_holder: StatusHolder):
        waited = 0.0
        while waited < self.waitTime:
            waitFor = min(self.waitTime - waited, 0.01)
            if status_holder.is_done.wait(waitFor):
                status_holder.status = "Cancelled"
                return
            waited += waitFor
            status_holder.status = "Waited for %.2f s" % (waited)
        status_holder.status = "End reached"


providers["Wait"] = _WaitTask
