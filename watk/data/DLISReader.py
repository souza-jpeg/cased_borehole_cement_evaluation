import os
import dlisio
import numpy as np

class DLISReader:
    """
    This class is responsible solely for extracting and standardizing curves from .DLIS files.
    Used to extract channels to WellCementDataset class.
    """

    DEFAULT_CHANNELS = [
        'CBL', 'CBLF', 'ECCE', 'AZEC', 'AWBK', 'IRAV', 'IRBK',
        'T2BK', 'AIBK', 'UFLG', 'UCAZ', 'RB', 'GR', 'VDL',
    ]

    _TO_METERS = {'m': 1.0, 'ft': 0.3048, 'in': 0.0254, '0.1 in': 0.00254}


    @staticmethod
    def load_well_curves(dlis_path: str, channels: list = None) -> dict:
        """
        Loads the curves from a DLIS file and returns a dictionary with the requested channels.
        """
        if not os.path.exists(dlis_path):
            raise FileNotFoundError(f"DLIS file not found: {dlis_path}")

        channels = channels or DLISReader.DEFAULT_CHANNELS

        dlisio.common.set_encodings(['latin1'])
        f, *f_tail = dlisio.dlis.load(dlis_path)

        curves_data = {}

        for frame in f.frames:
            frame_channels = [ch.name for ch in frame.channels]
            wanted = [c for c in channels if c in frame_channels and c not in curves_data]
            c_frame = frame.curves()
            depth = DLISReader._extract_depth(frame, c_frame)
            descending = len(depth) > 1 and depth[0] > depth[-1]
            if descending:
                depth = depth[::-1]

            for ch_name in wanted:
                data = np.nan_to_num(c_frame[ch_name], nan=0.0)
                if descending:
                    data = data[::-1]
                curves_data[ch_name] = data
                curves_data[f'depth_{ch_name}'] = depth

        missing = [c for c in channels if c not in curves_data]
        if missing:
            print(f"Warning: channels not found in any frame: {missing}")

        return curves_data


    @staticmethod
    def _extract_depth(frame, curves_dict) -> np.ndarray:
        """
        Extracts the depth vector and convert to meters.
        """
        depth_arr = np.asarray(curves_dict[frame.index]).copy()
        index_ch = next((ch for ch in frame.channels if ch.name == frame.index), None)

        units = getattr(index_ch, 'units', '') if index_ch else ''
        factor = DLISReader._TO_METERS.get(units)
        if factor is None:
            print(f"Warning: index unit '{units}' in frame '{frame.name}' "
                  f"is not recognized; assuming inches.")
            factor = 0.0254
        return depth_arr * factor
