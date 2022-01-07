import os
import json


class CurrentDir:
    def __init__(self, path):
        self.dir = '/'.join(path.split('/')[:-1])

    def __call__(self, subpath):
        return os.path.join(self.dir, subpath)


class Bunch(dict):
    def __init__(self, dictionary=dict(), **kwds):
        super().__init__(**kwds)
        self.__dict__ = self
        for key, val in dictionary.items():
            self.__dict__[key] = val
        for key, val in self.items():
            if isinstance(val, dict):
                self.__dict__[key] = Bunch(val)
            elif isinstance(val, list):
                for i in range(len(val)):
                    if isinstance(val[i], dict):
                        val[i] = Bunch(val[i])


def get_settings(curdir):
    with open(curdir('settings.json')) as f:
        settings = Bunch(json.load(f))
    return settings
