import os
import dlisio
import numpy as np

class DLISReader:
    """
    This class is responsible solely for extracting and standardizing curves from .DLIS files.
    Used to extract channels to WellCementDataset class.
    """
    @staticmethod
    def load_well_curves(dlis_path: str) -> dict:
        if not os.path.exists(dlis_path):
            raise FileNotFoundError(f"Arquivo DLIS não encontrado em: {dlis_path}")

        dlisio.common.set_encodings(['latin1'])
        f, *f_tail = dlisio.dlis.load(dlis_path)

        frame_60b = DLISReader._find_frame(f, target_channels=['CBL', 'AIBK'], fallback_name='60B')
        frame_20b = DLISReader._find_frame(f, target_channels=['VDL'], fallback_name='20B')

        curves_data = {}

        if frame_60b is not None:
            c_60 = frame_60b.curves()
            depth_60 = DLISReader._extract_depth(frame_60b, c_60)
            curves_data['depth_60'] = depth_60

            for ch_name in ['CBLF', 'ECCE', 'AZEC','AWBK', 'IRAV', 'IRBK', 'T2BK', 'AIBK', 'UFLG', 'UCAZ', 'RB', 'GR']:
                if ch_name in c_60.dtype.names:
                    curves_data[ch_name] = np.nan_to_num(c_60[ch_name], nan=0.0)

        if frame_20b is not None:
            c_20 = frame_20b.curves()
            depth_20 = DLISReader._extract_depth(frame_20b, c_20)
            curves_data['depth_20'] = depth_20

            if 'VDL' in c_20.dtype.names:
                curves_data['VDL'] = np.nan_to_num(c_20['VDL'], nan=0.0)

        return curves_data

    @staticmethod
    def _find_frame(dlis_obj, target_channels: list, fallback_name: str):
        """
        Find frame by channel names.
        TODO: Need to pass channels of interest as parameters?
        """
        for frame in dlis_obj.frames:
            chn_names = [ch.name for ch in frame.channels]
            if any(ch in chn_names for ch in target_channels):
                return frame
        try:
            return dlis_obj.object('FRAME', fallback_name)
        except Exception:
            return None

    @staticmethod
    def _extract_depth(frame, curves_dict) -> np.ndarray:
        """
        Extracts the depth vector and convert to meters.
        """
        depth_arr = curves_dict[frame.index].copy()
        index_ch = next((ch for ch in frame.channels if ch.name == frame.index), None)

        # convert 0.1 inches to meters (0.1 in * 0.0254 = 0.00254 m)
        if index_ch and getattr(index_ch, 'units', '') == '0.1 in':
            depth_arr *= 0.00254
        return depth_arr
