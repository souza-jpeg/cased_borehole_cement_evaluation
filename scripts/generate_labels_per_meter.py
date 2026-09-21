"""
Author: souza-jpeg

This script reads CSV files containing well data with 'Top' and 'Bottom' depth values,
and generates new CSV files where each row corresponds to a single meter of depth between
the 'Top' and 'Bottom' values. It also adds the corresponding well path from a predefined dictionary.

"""
import os
import pandas as pd

path, _ = os.path.split(os.path.abspath(__file__))
project_root = os.path.dirname(path)

labels_dir = os.path.join(project_root, "dataset", "labels")


well_files = {
    "F-9": "15_9-F-9_1.DLIS",
    "F-11": "15_9-F-11 B_2.DLIS",
    "F-12": "15_9-F-12_1.DLIS"
}


input_files = [
    "labels_classification_hydraulic_isolation.csv",
    "labels_classification_cement_quality.csv"
]

suffix = "_per_meter"

for file_name in input_files:
    input_path = os.path.join(labels_dir, file_name)
    print(f"Processando: {input_path}...")

    df = pd.read_csv(input_path)

    expanded_data = []

    for _, row in df.iterrows():
        top = int(row['Top'])
        bottom = int(row['Bottom'])
        well = row.get('Well', '')
        if well in well_files:
            path_str = f"dataset/dlis_data/{well_files[well]}"
        else:
            path_str = "Caminho não encontrado"
        other_cols = row.drop(['Top', 'Bottom']).to_dict()
        for depth_val in range(top, bottom + 1):
            new_row = other_cols.copy()
            new_row['Depth'] = float(depth_val)
            new_row['Path'] = path_str
            expanded_data.append(new_row)

    expanded_df = pd.DataFrame(expanded_data)

    cols = list(expanded_df.columns)
    if 'Well' in cols:
        cols.insert(1, cols.pop(cols.index('Depth')))
        cols.insert(2, cols.pop(cols.index('Path')))
    expanded_df = expanded_df[cols]

    output_filename = file_name.replace(".csv", f"{suffix}.csv")
    output_path = os.path.join(labels_dir, output_filename)
    expanded_df.to_csv(output_path, index=False)
    print(f"Concluído! Arquivo salvo em: {output_path}\n")
