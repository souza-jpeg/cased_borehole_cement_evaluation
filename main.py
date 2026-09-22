import os
import sys
from data_loader import WellDataset

output_dir = 'plots'

cement_csv_path = R'D:\cased_borehole_cement_evaluation\dataset\labels\labels_classification_cement_quality_per_meter.csv'
hydraulic_csv_path = R'D:\cased_borehole_cement_evaluation\dataset\labels\labels_classification_hydraulic_isolation_per_meter.csv'
path, _ = os.path.split(os.path.abspath(__file__))
project_root = R'D:\cased_borehole_cement_evaluation'


dataset = WellDataset(
    cq_csv_path=cement_csv_path,
    hi_csv_path=hydraulic_csv_path,
    project_root=project_root
)

dataset.describe(output_dir="plots")
