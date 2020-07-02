import json
import numpy as np
import inspect

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if inspect.isclass(type(obj)):
            return obj.__dict__()
        return json.JSONEncoder.default(self, obj)