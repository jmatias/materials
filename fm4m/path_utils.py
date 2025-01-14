class add_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        import sys

        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            import sys

            sys.path.remove(self.path)
        except ValueError:
            pass


def get_path_from_root(module: str, filename: str):
    import os
    import importlib.resources

    return os.path.abspath(str(importlib.resources.files(module).joinpath(filename)))
