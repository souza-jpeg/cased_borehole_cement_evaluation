import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_loader import WellDataset
from cement_evaluation_pytorch import CementEvaluationCNN, decode_ordinal_prediction

def decode_hi_prediction(pred_prob):
    return 1 if pred_prob >= 0.5 else 0

def evaluate_predictions(y_true, y_pred, task_type='bq'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    exact_matches = np.sum(y_true == y_pred)
    total = len(y_true)
    upa = (exact_matches / total) * 100 if total > 0 else 0.0

    if task_type == 'bq':
        adjacent_matches = np.sum(np.abs(y_true - y_pred) <= 1)
        uaa = (adjacent_matches / total) * 100 if total > 0 else 0.0
        return upa, uaa
    return upa, 0.0

def save_confusion_matrix(y_true, y_pred, classes, title, filepath):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)), normalize='true') * 100
    fig, ax = plt.subplots(figsize=(9, 7.5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap='magma', values_format='.1f', colorbar=True)

    # Ajuste de tamanho dos Títulos e Eixos
    plt.title(title, fontsize=25, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted label", fontsize=20, fontweight='bold', labelpad=10)
    ax.set_ylabel("True label", fontsize=20, fontweight='bold', labelpad=10)

    # Rotaciona os rótulos do eixo X em 45° na diagonal para não sobrepor
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=20, fontweight='bold')
    ax.set_yticklabels(classes, fontsize=20, fontweight='bold')

    # Aumenta os números/porcentagens dentro das células da matriz
    for text in disp.text_.ravel():
        text.set_fontsize(20)
        text.set_fontweight('bold')

    # Ajusta o tamanho dos números na barra de cores (colorbar)
    cbar = disp.ax_.images[-1].colorbar
    if cbar:
        cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_prediction_comparison(depths, true_labels, pred_labels, well_name, classes, title, filepath):
    fig, ax = plt.subplots(figsize=(4.5, 12))

    y_min, y_max = np.min(depths), np.max(depths)

    gt_matrix = np.array(true_labels).reshape(-1, 1)
    pr_matrix = np.array(pred_labels).reshape(-1, 1)
    combined = np.hstack([gt_matrix, pr_matrix])

    if len(classes) == 6:
        # Cores hexadecimais sólidas para Qualidade do Cimento
        hex_colors = ["#8B0000", "#FF8C00", "#FFE65C", "#ABF875", '#008000', '#006400']
        cmap = ListedColormap(hex_colors)
    else:
        # Isolamento Hidráulico
        hex_colors = ['#8B0000', '#006400']
        cmap = ListedColormap(hex_colors)

    cax = ax.imshow(
        combined, aspect='auto', interpolation='nearest',
        extent=[0, 2, y_max, y_min], cmap=cmap, vmin=-0.5, vmax=len(classes)-0.5
    )

    # Ajuste dos Ticks e Títulos com fontes maiores e rotação diagonal no eixo X
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Ground Truth', 'Prediction'], rotation=45, ha='right', fontsize=20, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=20, fontweight='bold')
    ax.set_title(f"{well_name}\n{title}", fontsize=25, fontweight='bold', pad=12)
    ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

def run_training_pipeline():
    result_dir = 'result'
    plots_dir = 'plots'
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    cement_csv = R'D:\cased_borehole_cement_evaluation\dataset\labels\labels_classification_cement_quality_per_meter.csv'
    hydraulic_csv = R'D:\cased_borehole_cement_evaluation\dataset\labels\labels_classification_hydraulic_isolation_per_meter.csv'
    project_root = R'D:\cased_borehole_cement_evaluation'

    print("loading dataset...")
    dataset = WellDataset(
        cq_csv_path=cement_csv,
        hi_csv_path=hydraulic_csv,
        project_root=project_root
    )

    print("describing dataset...")
    dataset.describe(output_dir=plots_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training in device: {device}\n")

    cq_classes = ["Free Pipe", "Poor", "Moderate to Poor", "Moderate", "Good to Moderate", "Good"]
    hi_classes = ["no", "yes"]

    print("==================================================")
    print(" STARTING TRAINING: CEMENT QUALITY (2-Fold)")
    print("==================================================")

    bq_wells = dataset.df.dropna(subset=[dataset.cq_label_col])['Well'].unique()
    bq_all_true, bq_all_pred = [], []
    bq_fold_upas, bq_fold_uaas = [], []

    for i, test_well in enumerate(bq_wells):
        train_wells = [w for w in bq_wells if w != test_well]
        print(f"\n[BQ Fold {i+1}] Training on {train_wells} | Testing on {test_well}")

        train_ds = torch.utils.data.ConcatDataset([dataset.get_pytorch_dataset(w, 'bq') for w in train_wells])
        test_ds = dataset.get_pytorch_dataset(test_well, 'bq')

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

        model = CementEvaluationCNN(num_classes=6).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(100):
            for usit, c1d, vdl, targets, _, _ in train_loader:
                usit, c1d, vdl, targets = usit.to(device), c1d.to(device), vdl.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(usit, c1d, vdl)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        model.eval()
        well_true, well_pred, well_depths = [], [], []
        with torch.no_grad():
            for usit, c1d, vdl, _, labels, depths in test_loader:
                usit, c1d, vdl = usit.to(device), c1d.to(device), vdl.to(device)
                outputs = model(usit, c1d, vdl).cpu().numpy()
                for idx_b in range(len(labels)):
                    pred_cls = decode_ordinal_prediction(outputs[idx_b])
                    true_cls = labels[idx_b].item()
                    well_true.append(true_cls)
                    well_pred.append(pred_cls)
                    well_depths.append(depths[idx_b].item())

        bq_all_true.extend(well_true)
        bq_all_pred.extend(well_pred)

        upa_fold, uaa_fold = evaluate_predictions(well_true, well_pred, 'bq')
        bq_fold_upas.append(upa_fold)
        bq_fold_uaas.append(uaa_fold)
        print(f" -> Test {test_well} | UPA: {upa_fold:.2f}% | UAA: {uaa_fold:.2f}%")

        plot_prediction_comparison(
            well_depths, well_true, well_pred, test_well, cq_classes,
            "Cement Quality (Test)", os.path.join(result_dir, f"cement_quality_{test_well}_comparison.png")
        )

    global_bq_upa, global_bq_uaa = evaluate_predictions(bq_all_true, bq_all_pred, 'bq')
    print(f"\n[Global BQ Result (Concatenado)] UPA: {global_bq_upa:.2f}% | UAA: {global_bq_uaa:.2f}%")
    print(f"[Global BQ Result (Média ± Std Folds)] UPA: {np.mean(bq_fold_upas):.2f}% ± {np.std(bq_fold_upas):.2f}% | UAA: {np.mean(bq_fold_uaas):.2f}% ± {np.std(bq_fold_uaas):.2f}%")

    save_confusion_matrix(
        bq_all_true, bq_all_pred, cq_classes,
        f"Cement Quality Confusion Matrix",
        os.path.join(result_dir, "confusion_matrix_cement_quality.png")
    )

    print("\n==================================================")
    print(" STARTING TRAINING: HYDRAULIC ISOLATION (3-Fold)")
    print("==================================================")

    hi_wells = dataset.df.dropna(subset=[dataset.hi_label_col])['Well'].unique()
    hi_all_true, hi_all_pred = [], []
    hi_fold_upas = []

    for i, test_well in enumerate(hi_wells):
        train_wells = [w for w in hi_wells if w != test_well]
        print(f"\n[HI Fold {i+1}] Training on {train_wells} | Testing on {test_well}")

        train_ds = torch.utils.data.ConcatDataset([dataset.get_pytorch_dataset(w, 'hi') for w in train_wells])
        test_ds = dataset.get_pytorch_dataset(test_well, 'hi')

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

        model = CementEvaluationCNN(num_classes=2).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        model.train()
        for epoch in range(100):
            for usit, c1d, vdl, targets, _, _ in train_loader:
                usit, c1d, vdl, targets = usit.to(device), c1d.to(device), vdl.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(usit, c1d, vdl)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        model.eval()
        well_true, well_pred, well_depths = [], [], []
        with torch.no_grad():
            for usit, c1d, vdl, _, labels, depths in test_loader:
                usit, c1d, vdl = usit.to(device), c1d.to(device), vdl.to(device)
                outputs = model(usit, c1d, vdl).cpu().numpy()
                for idx_b in range(len(labels)):
                    pred_cls = decode_hi_prediction(outputs[idx_b][0])
                    true_cls = labels[idx_b].item()
                    well_true.append(true_cls)
                    well_pred.append(pred_cls)
                    well_depths.append(depths[idx_b].item())

        hi_all_true.extend(well_true)
        hi_all_pred.extend(well_pred)

        upa_fold, _ = evaluate_predictions(well_true, well_pred, 'hi')
        hi_fold_upas.append(upa_fold)
        print(f" -> Test {test_well} | UPA (Precision): {upa_fold:.2f}%")

        plot_prediction_comparison(
            well_depths, well_true, well_pred, test_well, hi_classes,
            "Hydraulic Isolation (Test)", os.path.join(result_dir, f"hydraulic_isolation_{test_well}_comparison.png")
        )

    global_hi_upa, _ = evaluate_predictions(hi_all_true, hi_all_pred, 'hi')
    print(f"\n[Global HI Result (Concatenado)] UPA: {global_hi_upa:.2f}%")
    print(f"[Global HI Result (Média ± Std Folds)] UPA: {np.mean(hi_fold_upas):.2f}% ± {np.std(hi_fold_upas):.2f}%")

    save_confusion_matrix(
        hi_all_true, hi_all_pred, hi_classes,
        f"Hydraulic Isolation Confusion Matrix",
        os.path.join(result_dir, "confusion_matrix_hydraulic_isolation.png")
    )
    print(f"\nTraining completed successfully! All results and plots were saved in the folder '{result_dir}'.")

if __name__ == '__main__':
    run_training_pipeline()
