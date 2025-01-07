import importlib.resources
import os
import sys


class add_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


def get_path_from_root(module: str, filename: str):
    return os.path.abspath(str(importlib.resources.files(module).joinpath(filename)))
