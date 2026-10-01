import os
import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

from watk.data.DLISReader import DLISReader

class WellCementDataset(Dataset):
    """
    Dataset for load DLIS wellbore data for 3 branches Viggen's neural network.
    """
    def __init__(
            self,
            cq_csv_path: str,
            hi_csv_path: str,
            project_root: str = "",
            window_size: int = 171
        ):
        self.project_root = project_root
        self.window_size = window_size

        # TODO: it worth to parameterize the classes?
        self.cq_classes = ["Free Pipe", "Poor", "Moderate to Poor", "Moderate", "Good to Moderate", "Good"]
        self.hi_classes = ["no", "yes"]

        self.df = self._load_and_merge_csvs(cq_csv_path, hi_csv_path)

        self.well_cache = {}
        self._preload_dlis_files()

        self.samples = self._build_samples()

    def _load_and_merge_csvs(self, cq_path: str, hi_path: str) -> pd.DataFrame:
        df_cq = pd.read_csv(cq_path)
        df_hi = pd.read_csv(hi_path)

        col_cq = [c for c in df_cq.columns if c not in ['Well', 'Depth', 'Path']][0]
        col_hi = [c for c in df_hi.columns if c not in ['Well', 'Depth', 'Path']][0]
        df_cq = df_cq.rename(columns={col_cq: 'Cement_Quality'})
        df_hi = df_hi.rename(columns={col_hi: 'Hydraulic_Isolation'})

        df_merged = pd.merge(df_cq, df_hi[['Well', 'Depth', 'Path', 'Hydraulic_Isolation']],
                             on=['Well', 'Depth'], how='outer', suffixes=('_cq', '_hi'))

        if 'Path_cq' in df_merged.columns and 'Path_hi' in df_merged.columns:
            df_merged['Path'] = df_merged['Path_cq'].fillna(df_merged['Path_hi'])
            df_merged = df_merged.drop(columns=['Path_cq', 'Path_hi'])

        well_order = pd.unique(pd.concat([df_cq['Well'], df_hi['Well']]))
        well_rank = {w: i for i, w in enumerate(well_order)}

        df_merged = (
            df_merged
            .assign(_well_rank=df_merged['Well'].map(well_rank))
            .sort_values(['_well_rank', 'Depth'])
            .drop(columns='_well_rank')
            .reset_index(drop=True)
        )

        return df_merged

    def _preload_dlis_files(self):
        """
        Reads .DLIS data for each well and store them in self.well_cache.
        """
        unique_wells = self.df['Well'].unique()
        print(f"\n[Dataset] Pré-loaded data for {len(unique_wells)} well(s)...")

        for well in unique_wells:
            df_w = self.df[self.df['Well'] == well]
            raw_path = str(df_w['Path'].iloc[0])

            parts = [p for p in re.split(r'[\\/]', raw_path) if p]
            dlis_full_path = os.path.abspath(os.path.join(self.project_root, *parts)) if self.project_root else os.path.abspath(os.path.join(*parts))

            print(f" -> Loading DLIS data for well '{well}': {dlis_full_path}")
            try:
                self.well_cache[well] = DLISReader.load_well_curves(dlis_full_path)
            except Exception as e:
                print(f"    [Error] Error reading DLIS for well {well}: {e}. Using none data as fallback.")
                self.well_cache[well] = None

    def _build_samples(self) -> list:
        """
        Builds the list of samples by linking the depth from the CSV with the indices from the DLIS.
        """
        samples = []

        for idx, row in self.df.iterrows():
            well = row['Well']
            target_depth = row['Depth']
            curves = self.well_cache.get(well)

            cq_label = self.cq_classes.index(str(row['Cement_Quality'])) if str(row['Cement_Quality']) in self.cq_classes else 0
            hi_label = self.hi_classes.index(str(row['Hydraulic_Isolation'])) if str(row['Hydraulic_Isolation']) in self.hi_classes else 0

            samples.append({
                'well': well,
                'depth': target_depth,
                'cq_label': cq_label,
                'hi_label': hi_label,
                'curves': curves,
                'csv_idx': idx
            })

        print(f"[Dataset] Total: {len(samples)} samples per meter ready in memory.\n")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        curves = sample['curves']
        target_depth = sample['depth']

        if curves is not None and 'depth_60' in curves:
            depth_arr = curves['depth_60']
            center_idx = int(np.searchsorted(depth_arr, target_depth))

            start_idx = max(0, center_idx - self.window_size // 2)
            end_idx = min(len(depth_arr), start_idx + self.window_size)

            if 'AIBK' in curves:
                usit_seg = curves['AIBK'][start_idx:end_idx].T # Shape (72, N)
            else:
                usit_seg = np.zeros((72, self.window_size))

            cbl_val = curves['CBL'][center_idx] if ('CBL' in curves and center_idx < len(curves['CBL'])) else 0.0
        else:
            usit_seg = np.zeros((72, self.window_size))
            cbl_val = 0.0

        if usit_seg.shape[1] < self.window_size:
            pad_len = self.window_size - usit_seg.shape[1]
            usit_seg = np.pad(usit_seg, ((0, 0), (0, pad_len)))

        usit_norm = (usit_seg - usit_seg.mean()) / (usit_seg.std() + 1e-6)

        return {
            'usit': torch.tensor(usit_norm, dtype=torch.float32),
            'cbl': torch.tensor(cbl_val, dtype=torch.float32),
            'cq_label': sample['cq_label'],
            'hi_label': sample['hi_label'],
            'depth': sample['depth'],
            'well': sample['well']
        }
