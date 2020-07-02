import audioread.maddec
import numpy as np
import mad
import librosa
from audioread.maddec import UnsupportedError

def load_audio(path, sr=22050, mono=True, offset=0.0, duration=None, dtype=np.float32, res_type='kaiser_best'):
    y, sr_native = __audioread_load(path, offset, duration, dtype)
    if mono:
        y = librosa.to_mono(y)
    if sr is not None:
        y = librosa.resample(y, sr_native, sr, res_type=res_type)
    else:
        sr = sr_native
    return y, sr


class MadAudioFile(object):
    """MPEG audio file decoder using the MAD library."""
    def __init__(self, filename):
        self.fp = open(filename, 'rb')
        self.mf = mad.MadFile(self.fp)
        if not self.mf.total_time():  # Indicates a failed open.
            self.fp.close()
            raise UnsupportedError()

    def close(self):
        if hasattr(self, 'fp'):
            self.fp.close()
        if hasattr(self, 'mf'):
            del self.mf

    def read_blocks(self, block_size=4096):
        """Generates buffers containing PCM data for the audio file.
        """
        while True:
            out = self.mf.read(block_size)
            if not out:
                break
            yield bytes(out)

    @property
    def samplerate(self):
        """Sample rate in Hz."""
        return self.mf.samplerate()

    @property
    def duration(self):
        """Length of the audio in seconds (a float)."""
        return float(self.mf.total_time()) / 1000

    @property
    def channels(self):
        """The number of channels."""
        if self.mf.mode() == mad.MODE_SINGLE_CHANNEL:
            return 1
        elif self.mf.mode() in (mad.MODE_DUAL_CHANNEL,
                                mad.MODE_JOINT_STEREO,
                                mad.MODE_STEREO):
            return 2
        else:
            # Other mode?
            return 2

    def __del__(self):
        self.close()

    # Iteration.
    def __iter__(self):
        return self.read_blocks()

    # Context manager.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False



def _convert_buffer_to_float(buf, n_bytes=2, dtype=np.float32):
    # taken from librosa.util.utils
    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))
    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)
    # Rescale and format the data buffer
    out = scale * np.frombuffer(buf, fmt).astype(dtype)
    return out

# def load_ar_mad(fp):
#     with audioread.maddec.MadAudioFile(fp) as f:
#         total_frames = 0
#         for buf in f:
#             sig = _convert_buffer_to_float(buf)
#             sig = sig.reshape(f.channels, -1)
#             total_frames += sig.shape[1]
#         return sig



def __audioread_load(path, offset, duration, dtype):
    '''Load an audio buffer using audioread.
    This loads one block at a time, and then concatenates the results.
    '''

    y = []
    with MadAudioFile(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))
                               * n_channels)

        n = 0

        for frame in input_file:
            frame = _convert_buffer_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, sr_native
