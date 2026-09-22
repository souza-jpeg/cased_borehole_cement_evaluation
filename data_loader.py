import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import dlisio

class WellDataset:
    def __init__(self, cq_csv_path, hi_csv_path, project_root=""):
        self.project_root = project_root

        print("Loading CSV files...")
        self.df_cq = pd.read_csv(cq_csv_path)
        self.df_hi = pd.read_csv(hi_csv_path)

        cq_col_original = [col for col in self.df_cq.columns if col not in ['Well', 'Depth', 'Path']][0]
        hi_col_original = [col for col in self.df_hi.columns if col not in ['Well', 'Depth', 'Path']][0]

        self.cq_label_col = "Cement_Quality"
        self.hi_label_col = "Hydraulic_Isolation"

        self.df_cq = self.df_cq.rename(columns={cq_col_original: self.cq_label_col})
        self.df_hi = self.df_hi.rename(columns={hi_col_original: self.hi_label_col})

        self.df = pd.merge(
            self.df_cq,
            self.df_hi[['Well', 'Depth', 'Path', self.hi_label_col]],
            on=['Well', 'Depth'],
            how='outer'
        )

        if 'Path_x' in self.df.columns and 'Path_y' in self.df.columns:
            self.df['Path'] = self.df['Path_x'].fillna(self.df['Path_y'])
            self.df = self.df.drop(columns=['Path_x', 'Path_y'])

        print(f"Dataset loaded with {len(self.df)} total records from {self.df['Well'].nunique()} wells.")

    def _extract_dlis_data(self, dlis_path):
        try:
            dlisio.common.set_encodings(['latin1'])
            f, *f_tail = dlisio.dlis.load(dlis_path)
        except Exception as e:
            print(f"Error loading DLIS {dlis_path}: {e}")
            return []

        frame_60b, frame_20b = None, None
        for frame in f.frames:
            channel_names = [ch.name for ch in frame.channels]
            if 'CBL' in channel_names or 'AIBK' in channel_names:
                frame_60b = frame
            if 'VDL' in channel_names:
                frame_20b = frame

        if frame_60b is None:
            try: frame_60b = f.object('FRAME', '60B')
            except: pass
        if frame_20b is None:
            try: frame_20b = f.object('FRAME', '20B')
            except: pass

        available_tracks = []

        if frame_60b is not None:
            curves_60b = frame_60b.curves()
            index_ch_60b = next((ch for ch in frame_60b.channels if ch.name == frame_60b.index), None)
            depth_60b = curves_60b[frame_60b.index].copy()
            if index_ch_60b and index_ch_60b.units == '0.1 in':
                depth_60b *= 0.00254

            chns_60b = [ch.name for ch in frame_60b.channels]
            for ch in ['CBL', 'AIBK', 'IRBK', 'T2BK', 'ERBK']:
                if ch in chns_60b:
                    available_tracks.append((ch, curves_60b, depth_60b))

        if frame_20b is not None:
            curves_20b = frame_20b.curves()
            index_ch_20b = next((ch for ch in frame_20b.channels if ch.name == frame_20b.index), None)
            depth_20b = curves_20b[frame_20b.index].copy()
            if index_ch_20b and index_ch_20b.units == '0.1 in':
                depth_20b *= 0.00254

            chns_20b = [ch.name for ch in frame_20b.channels]
            if 'VDL' in chns_20b:
                available_tracks.append(('VDL', curves_20b, depth_20b))

        return available_tracks

    def describe(self, output_dir="plots_describe"):
        os.makedirs(output_dir, exist_ok=True)

        cmap_cq = ListedColormap(['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen'])
        cmap_hi = ListedColormap(['darkred', 'darkgreen'])

        hist_cq = {}
        hist_hi = {}

        cq_classes = ["Free Pipe", "Poor", "Moderate to Poor", "Moderate", "Good to Moderate", "Good"]
        hi_classes = ["no", "yes"]

        wells = self.df['Well'].unique()

        for well in wells:
            print(f"Processing well: {well}...")
            df_well = self.df[self.df['Well'] == well].sort_values('Depth')

            dlis_path_raw = str(df_well['Path'].iloc[0])
            path_parts = [p for p in re.split(r'[\\/]', dlis_path_raw) if p]

            if self.project_root:
                dlis_path = os.path.abspath(os.path.join(self.project_root, *path_parts))
            else:
                dlis_path = os.path.abspath(os.path.join(*path_parts))

            hist_cq[well] = df_well[self.cq_label_col].value_counts().to_dict()
            hist_hi[well] = df_well[self.hi_label_col].value_counts().to_dict()

            if not os.path.exists(dlis_path):
                print(f"  -> Warning: DLIS not found at {dlis_path}. Skipping profile plot.")
                continue

            available_tracks = self._extract_dlis_data(dlis_path)
            if not available_tracks:
                print("  -> No compatible DLIS channels found. Skipping.")
                continue

            num_tracks = len(available_tracks) + 2

            fig, axes = plt.subplots(1, num_tracks, figsize=(1.6 * num_tracks, 16), sharey=True)
            fig.suptitle(f"Well {well} - Profiles and Labels", fontsize=14, y=0.995)

            global_min_d = min(np.min(t[2]) for t in available_tracks)
            global_max_d = max(np.max(t[2]) for t in available_tracks)

            for i, (track_name, curves_dict, depth_arr) in enumerate(available_tracks):
                ax = axes[i]
                max_d, min_d = np.max(depth_arr), np.min(depth_arr)

                if track_name == 'CBL':
                    ax.plot(curves_dict['CBL'], depth_arr, color='black', linewidth=0.5)
                    ax.set_title("CBL\n(0 mV 53)", pad=10)
                    ax.set_xlim(0, 53)
                elif track_name == 'AIBK':
                    ax.imshow(curves_dict['AIBK'], aspect='auto', cmap='YlOrBr', vmin=0, vmax=7.5, extent=[0, 360, min_d, max_d])
                    ax.set_title("AIBK\n(MRayl)", pad=10)
                elif track_name == 'IRBK':
                    ax.imshow(curves_dict['IRBK'], aspect='auto', cmap='seismic', vmin=-0.045, vmax=0.045, extent=[0, 360, min_d, max_d])
                    ax.set_title("IRBK\n(in)", pad=10)
                elif track_name == 'T2BK':
                    ax.imshow(curves_dict['T2BK'], aspect='auto', cmap='seismic', vmin=-0.035, vmax=0.035, extent=[0, 360, min_d, max_d])
                    ax.set_title("T2BK\n(in)", pad=10)
                elif track_name == 'ERBK':
                    ax.imshow(curves_dict['ERBK'], aspect='auto', cmap='viridis', vmin=4.83, vmax=4.91, extent=[0, 360, min_d, max_d])
                    ax.set_title("ERBK\n(in)", pad=10)
                elif track_name == 'VDL':
                    vdl = np.nan_to_num(curves_dict['VDL'], nan=0.0)
                    ax.imshow(vdl, aspect='auto', cmap='gray', extent=[200, 1200, min_d, max_d], vmin=np.percentile(vdl, 5), vmax=np.percentile(vdl, 95))
                    ax.set_title("VDL\n(200 us 1200)", pad=10)
                    ax.set_xlim(200, 1200)

                ax.set_xticks([])
                ax.grid(False)

            ax_cq = axes[-2]
            ax_hi = axes[-1]

            min_csv_d = int(np.floor(df_well['Depth'].min()))
            max_csv_d = int(np.ceil(df_well['Depth'].max()))

            dense_edges = np.arange(min_csv_d, max_csv_d + 2)

            z_cq_dense = np.full(max_csv_d - min_csv_d + 1, np.nan)
            z_hi_dense = np.full(max_csv_d - min_csv_d + 1, np.nan)

            valid_indices = (df_well['Depth'] - min_csv_d).astype(int).values

            def map_cq(x):
                val = str(x)
                return cq_classes.index(val) if val in cq_classes else np.nan

            def map_hi(x):
                val = str(x)
                return hi_classes.index(val) if val in hi_classes else np.nan

            z_cq_dense[valid_indices] = df_well[self.cq_label_col].apply(map_cq).values
            z_hi_dense[valid_indices] = df_well[self.hi_label_col].apply(map_hi).values

            z_cq_dense = z_cq_dense.reshape(-1, 1)
            z_hi_dense = z_hi_dense.reshape(-1, 1)

            ax_cq.pcolormesh([0, 1], dense_edges, z_cq_dense, cmap=cmap_cq, vmin=0, vmax=5)
            ax_cq.set_title("Cement\nQuality", pad=10)
            ax_cq.set_xticks([])

            ax_hi.pcolormesh([0, 1], dense_edges, z_hi_dense, cmap=cmap_hi, vmin=0, vmax=1)
            ax_hi.set_title("Hydraulic\nIsolation", pad=10)
            ax_hi.set_xticks([])

            axes[0].set_ylabel('Depth (m)')
            min_tick = np.ceil(global_min_d / 10) * 10
            max_tick = np.floor(global_max_d / 10) * 10
            depth_ticks = np.arange(min_tick, max_tick + 1, 10)
            axes[0].set_yticks(depth_ticks)
            axes[0].set_yticklabels([f"{t:.1f}" for t in depth_ticks])
            for ax in axes[1:]:
                ax.tick_params(left=False)

            axes[0].invert_yaxis()

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.subplots_adjust(top=0.90, wspace=0.15)

            plot_path = os.path.join(output_dir, f"{well}_profiles_and_labels.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  -> Plot saved at: {plot_path}")

        self._plot_histograms(hist_cq, hist_hi, cq_classes, hi_classes, output_dir)
        print("\nDescription process completed successfully!")

    def _plot_histograms(self, hist_cq, hist_hi, cq_classes, hi_classes, output_dir):
        print("Generating class distribution histograms...")

        df_hist_cq = pd.DataFrame(hist_cq).T.fillna(0)
        for c in cq_classes:
            if c not in df_hist_cq.columns:
                df_hist_cq[c] = 0
        df_hist_cq = df_hist_cq[cq_classes]

        df_hist_hi = pd.DataFrame(hist_hi).T.fillna(0)
        for c in hi_classes:
            if c not in df_hist_hi.columns:
                df_hist_hi[c] = 0
        df_hist_hi = df_hist_hi[hi_classes]

        fig, axes = plt.subplots(2, 1, figsize=(14, 12))

        cq_colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
        df_hist_cq.plot(kind='bar', stacked=True, color=cq_colors, ax=axes[0], edgecolor='black')
        axes[0].set_title('Class Distribution - Cement Quality per Well')
        axes[0].set_ylabel('Meters (Quantity)')
        axes[0].legend(title="Classes", bbox_to_anchor=(1.01, 1), loc='upper left')
        axes[0].tick_params(axis='x', rotation=0)

        hi_colors = ['darkred', 'darkgreen']
        df_hist_hi.plot(kind='bar', stacked=True, color=hi_colors, ax=axes[1], edgecolor='black')
        axes[1].set_title('Class Distribution - Hydraulic Isolation per Well')
        axes[1].set_ylabel('Meters (Quantity)')
        axes[1].legend(title="Classes", bbox_to_anchor=(1.01, 1), loc='upper left')
        axes[1].tick_params(axis='x', rotation=0)

        plt.tight_layout()
        hist_path = os.path.join(output_dir, "classes_histogram_per_well.png")
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Final histogram saved at: {hist_path}")
