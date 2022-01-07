import os
import json
import numpy as np
from scipy.signal import savgol_filter
from .misc import Bunch


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj,
                      (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                       np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray, )):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


class SummaryWriter:
    def __init__(self, data_store_location='runs', reset=False):
        self.store_location = data_store_location + '.json'
        if not os.path.exists(self.store_location) or reset:
            self.data = Bunch({'vals': {}, 'smoothing': 0.6})
        else:
            with open(self.store_location) as f:
                self.data = Bunch(json.load(f))
        self.save_every = 1000
        self.save_count = 0

    def set_smoothing(self, smoothing, tag=None):
        if tag is not None:
            self.data.vals[tag].smoothing = smoothing
        else:
            self.data.smoothing = smoothing

    def add_scalar(self, tag, value, global_step=-1):
        if type(value) == np.int64:
            value = int(value)
        if type(value) == np.float64:
            value = float(value)
        assert type(tag) == str, 'tag must be str'
        assert type(value) in (
            int, float), f'value type not supported: {type(value)}'
        assert type(global_step) == int, 'global_step should be int'
        if tag not in self.data.vals:
            self.data.vals[tag] = Bunch({'scalar': []})
        self.data.vals[tag].scalar.append(
            (global_step
             if global_step != -1 else len(self.data.vals[tag].scalar), value))
        self.save_count += 1
        if self.save_count > self.save_every:
            self.flush()
            self.save_count = 0

    def add_plot(self):
        pass

    def plot(self, imgs_location='imgs'):
        self.flush()
        import plotly
        import plotly.graph_objects as go

        plotly.io.orca.config.executable = '/home/bluesk/anaconda3/bin/orca'
        if not os.path.exists(imgs_location):
            os.mkdir(imgs_location)
        for key in self.data.vals:
            scalar = self.data.vals[key].scalar
            sorted_values = np.array(sorted(scalar,
                                            key=lambda x: x[0])).T.tolist()
            smoothing = self.data.vals[key].get('smoothing',
                                                self.data.smoothing)
            window = 60 * smoothing
            window = int(window + 1) if window % 2 == 0 else int(window)
            if 6 * window < len(sorted_values[1]):
                sorted_values[1] = savgol_filter(sorted_values[1], window, 3)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sorted_values[0], y=sorted_values[1]))
            fig.write_image(os.path.join(imgs_location, key + '.png'))

    def flush(self):
        try:
            json_str = json.dumps(self.data, cls=NumpyEncoder)
        except Exception as e:
            print(e)
        else:
            with open(self.store_location, 'w') as f:
                f.write(json_str)


def main():
    writer = SummaryWriter()
    x = np.arange(10)
    for v in x:
        writer.add_scalar('x', v**2)

    writer.plot()


if __name__ == '__main__':
    main()
