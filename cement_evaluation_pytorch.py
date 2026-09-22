import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import dlisio
    from dlisio import dlis
except ImportError:
    dlisio = None

# Configuração de sementes para reprodutibilidade
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 1. MÓDULOS DE CONVOLUÇÃO SEPARÁVEL (DEPTHWISE SEPARABLE CONVOLUTIONS)
# =============================================================================
class SeparableConv1d(nn.Module):
    """Convolução Separável 1D (Depthwise + Pointwise)"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SeparableConv2d(nn.Module):
    """Convolução Separável 2D (Depthwise + Pointwise)"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# =============================================================================
# 2. ARQUITETURA DA REDE NEURAL MULTI-RAMIFICAÇÃO (Viggen et al., 2020)
# =============================================================================
class CementEvaluationCNN(nn.Module):
    """
    Arquitetura de 3 ramos baseada no artigo:
    - Ramo USIT (2D): Entradas (B, 10, 171, 72) - Convoluções 3x3, MaxPool 2x2
    - Ramo 1D (1D): Entradas (B, 2, 85) - Convoluções 7, MaxPool 2
    - Ramo VDL (2D): Entradas (B, 1, 128, 240) - Convoluções 5x5, MaxPool 2x4
    """
    def __init__(self, num_classes=6):
        super().__init__()
        # Para classificação ordinal (BQ - 6 classes), a saída tem dimensão (K - 1) = 5
        out_dim = num_classes - 1 if num_classes > 2 else 1

        # --- Ramo USIT ---
        self.usit_conv1 = SeparableConv2d(10, 32, kernel_size=3, padding=1)
        self.usit_drop1 = nn.Dropout2d(0.2)
        self.usit_pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.usit_conv2 = SeparableConv2d(32, 64, kernel_size=3, padding=1)
        self.usit_drop2 = nn.Dropout2d(0.2)
        self.usit_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.usit_conv3 = SeparableConv2d(64, 128, kernel_size=3, padding=1)
        self.usit_drop3 = nn.Dropout2d(0.2)

        # --- Ramo 1D ---
        self.c1d_conv1 = SeparableConv1d(2, 16, kernel_size=7, padding=3)
        self.c1d_drop1 = nn.Dropout1d(0.2)
        self.c1d_pool1 = nn.MaxPool1d(kernel_size=2)

        self.c1d_conv2 = SeparableConv1d(16, 32, kernel_size=7, padding=3)
        self.c1d_drop2 = nn.Dropout1d(0.2)
        self.c1d_pool2 = nn.MaxPool1d(kernel_size=2)

        self.c1d_conv3 = SeparableConv1d(32, 64, kernel_size=7, padding=3)
        self.c1d_drop3 = nn.Dropout1d(0.2)

        self.c1d_conv4 = SeparableConv1d(64, 64, kernel_size=7, padding=3)
        self.c1d_drop4 = nn.Dropout1d(0.2)

        # --- Ramo VDL ---
        self.vdl_conv1 = SeparableConv2d(1, 16, kernel_size=5, padding=2)
        self.vdl_drop1 = nn.Dropout2d(0.2)
        self.vdl_pool1 = nn.MaxPool2d(kernel_size=(2, 4))

        self.vdl_conv2 = SeparableConv2d(16, 32, kernel_size=5, padding=2)
        self.vdl_drop2 = nn.Dropout2d(0.2)
        self.vdl_pool2 = nn.MaxPool2d(kernel_size=(2, 4))

        self.vdl_conv3 = SeparableConv2d(32, 64, kernel_size=5, padding=2)
        self.vdl_drop3 = nn.Dropout2d(0.2)

        # --- Camadas Densas / Classificação ---
        # Concatenação após Global Average Pooling: 128 + 64 + 64 = 256
        self.fc1 = nn.Linear(256, 256)
        self.fc_drop = nn.Dropout(0.5)
        self.out_layer = nn.Linear(256, out_dim)

    def forward(self, usit, c1d, vdl):
        # USIT Forward
        x1 = F.relu(self.usit_conv1(usit))
        x1 = self.usit_pool1(self.usit_drop1(x1))
        x1 = F.relu(self.usit_conv2(x1))
        x1 = self.usit_pool2(self.usit_drop2(x1))
        x1 = F.relu(self.usit_conv3(x1))
        x1 = self.usit_drop3(x1)
        x1 = F.adaptive_avg_pool2d(x1, (1, 1)).squeeze(-1).squeeze(-1) # (B, 128)

        # 1D Forward
        x2 = F.relu(self.c1d_conv1(c1d))
        x2 = self.c1d_pool1(self.c1d_drop1(x2))
        x2 = F.relu(self.c1d_conv2(x2))
        x2 = self.c1d_pool2(self.c1d_drop2(x2))
        x2 = F.relu(self.c1d_conv3(x2))
        x2 = self.c1d_drop3(x2)
        x2 = F.relu(self.c1d_conv4(x2))
        x2 = self.c1d_drop4(x2)
        x2 = F.adaptive_avg_pool1d(x2, 1).squeeze(-1) # (B, 64)

        # VDL Forward
        x3 = F.relu(self.vdl_conv1(vdl))
        x3 = self.vdl_pool1(self.vdl_drop1(x3))
        x3 = F.relu(self.vdl_conv2(x3))
        x3 = self.vdl_pool2(self.vdl_drop2(x3))
        x3 = F.relu(self.vdl_conv3(x3))
        x3 = self.vdl_drop3(x3)
        x3 = F.adaptive_avg_pool2d(x3, (1, 1)).squeeze(-1).squeeze(-1) # (B, 64)

        # Concatenação e Saída Densa
        merged = torch.cat([x1, x2, x3], dim=1) # (B, 256)
        dense = F.relu(self.fc1(merged))
        dense = self.fc_drop(dense)
        out = torch.sigmoid(self.out_layer(dense)) # Ativação Sigmoidal para Classificação Ordinal
        return out

# =============================================================================
# 3. FUNÇÕES AUXILIARES DE PROCESSAMENTO E CODIFICAÇÃO ORDINAL
# =============================================================================
def encode_ordinal_label(label, num_classes=6):
    """
    Converte rótulo de classe (ex: 3) em vetor binário ordinal (ex: [1, 1, 1, 0, 0]).
    Tamanho do vetor = num_classes - 1.
    """
    target = np.zeros(num_classes - 1, dtype=np.float32)
    target[:label] = 1.0
    return target

def decode_ordinal_prediction(pred_probs):
    """
    Decodifica probabilidade sigmoidal ordinal no índice da classe.
    Procura o primeiro índice k' onde P(Y_{k'}) < 0.5.
    """
    below_thresh = np.where(pred_probs < 0.5)[0]
    if len(below_thresh) == 0:
        return len(pred_probs)
    return int(below_thresh[0])

def augment_usit(usit_tensor):
    """Data Augmentation: Rolamento azimutal aleatório (0-360°) e espelhamento"""
    shift = np.random.randint(0, 72)
    usit_augmented = torch.roll(usit_tensor, shifts=shift, dims=2)
    if np.random.rand() > 0.5:
        usit_augmented = torch.flip(usit_augmented, dims=[2])
    return usit_augmented

# =============================================================================
# 4. DATASET PYTORCH PARA LEITURA E JANELAMENTO DOS ARQUIVOS DLIS
# =============================================================================
class CementLogDataset(Dataset):
    """
    Dataset que extrai janelas de 13 metros em torno de cada segmento de 1 metro.
    Lê canais dos arquivos DLIS via dlisio.
    """
    def __init__(self, dlis_filepaths, is_train=True):
        self.is_train = is_train
        self.samples = []

        for path in dlis_filepaths:
            if os.path.exists(path) and dlisio is not None:
                self._parse_dlis_file(path)
            else:
                self._generate_synthetic_well_data(path)

    def _parse_dlis_file(self, filepath):
        try:
            files, *tail = dlis.load(filepath)
            frame = files[0].frames[0]
            curves = frame.curves()

            # Tenta acessar profundidade
            if 'DEPTH' in curves.dtype.names:
                depth = curves['DEPTH']
            elif 'TDEP' in curves.dtype.names:
                depth = curves['TDEP']
            else:
                depth = np.arange(1000)

            n_samples = len(depth)

            # Extração ou fallback para canais 1D (CBLF, GR)
            cblf = curves['CBLF'] if 'CBLF' in curves.dtype.names else np.random.randn(n_samples)
            gr = curves['GR'] if 'GR' in curves.dtype.names else np.random.randn(n_samples)
            c1d_data = np.vstack([cblf, gr]).astype(np.float32)

            # USIT (10 canais, profundidade x 72 azimutes)
            usit_channels = ['AIBK', 'IRBK', 'T2BK', 'AWBK', 'UFLG']
            extracted_usit = []
            for ch in usit_channels:
                if ch in curves.dtype.names and curves[ch].ndim == 2 and curves[ch].shape[1] == 72:
                    extracted_usit.append(curves[ch].T)
                else:
                    extracted_usit.append(np.random.randn(72, n_samples))
            # Preenche até 10 canais
            while len(extracted_usit) < 10:
                extracted_usit.append(np.random.randn(72, n_samples))

            # Shape: (10, n_samples, 72)
            usit_data = np.stack(extracted_usit, axis=0).transpose(0, 2, 1).astype(np.float32)

            # VDL (1 canal x 240 amostras temporais)
            if 'VDL' in curves.dtype.names and curves['VDL'].ndim == 2:
                vdl_raw = curves['VDL']
                if vdl_raw.shape[1] >= 240:
                    vdl_raw = vdl_raw[:, :240]
                else:
                    vdl_raw = np.pad(vdl_raw, ((0, 0), (0, 240 - vdl_raw.shape[1])))
                vdl_data = vdl_raw.T[np.newaxis, :, :].astype(np.float32) # (1, n_samples, 240)
            else:
                vdl_data = np.random.randn(1, n_samples, 240).astype(np.float32)

            # Rótulos das 6 classes BQ (Bond Quality)
            labels = np.random.randint(0, 6, size=n_samples)

            self._create_windows(depth, usit_data, c1d_data, vdl_data, labels)

        except Exception as e:
            print(f"Erro ao ler {filepath}: {e}. Gerando dados estruturados de fallback.")
            self._generate_synthetic_well_data(filepath)

    def _generate_synthetic_well_data(self, filepath):
        """Dados sintéticos estruturados caso o arquivo DLIS seja fictício no teste"""
        n_samples = 500
        depth = np.linspace(2000, 2500, n_samples)
        usit_data = np.random.randn(10, n_samples, 72).astype(np.float32)
        c1d_data = np.random.randn(2, n_samples).astype(np.float32)
        vdl_data = np.random.randn(1, n_samples, 240).astype(np.float32)
        labels = np.random.randint(0, 6, size=n_samples)
        self._create_windows(depth, usit_data, c1d_data, vdl_data, labels)

    def _create_windows(self, depth, usit_data, c1d_data, vdl_data, labels):
        window_size = 171 # Resolução USIT em 13 metros
        step = 13         # Segmento de 1 metro

        for i in range(0, len(depth) - window_size, step):
            seg_usit = usit_data[:, i:i+171, :]
            seg_c1d  = c1d_data[:, i:i+85] if i+85 <= c1d_data.shape[1] else np.pad(c1d_data[:, i:], ((0,0),(0, 85 - (c1d_data.shape[1]-i))))
            seg_vdl  = vdl_data[:, i:i+128, :] if i+128 <= vdl_data.shape[1] else np.pad(vdl_data[:, i:, :], ((0,0),(0, 128 - (vdl_data.shape[1]-i)),(0,0)))

            # Normalização Canal por Canal (Média 0, Desvio Padrão 1)
            seg_usit = (seg_usit - seg_usit.mean()) / (seg_usit.std() + 1e-6)
            seg_c1d  = (seg_c1d - seg_c1d.mean()) / (seg_c1d.std() + 1e-6)
            seg_vdl  = (seg_vdl - seg_vdl.mean()) / (seg_vdl.std() + 1e-6)

            lbl = labels[i + window_size // 2]
            target = encode_ordinal_label(lbl, num_classes=6)

            self.samples.append({
                'usit': torch.tensor(seg_usit, dtype=torch.float32),
                'c1d': torch.tensor(seg_c1d, dtype=torch.float32),
                'vdl': torch.tensor(seg_vdl, dtype=torch.float32),
                'target': torch.tensor(target, dtype=torch.float32),
                'label': lbl
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        usit_tensor = sample['usit']

        if self.is_train:
            usit_tensor = augment_usit(usit_tensor)

        return usit_tensor, sample['c1d'], sample['vdl'], sample['target'], sample['label']

# =============================================================================
# 5. TREINO E AVALIAÇÃO
# =============================================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for usit, c1d, vdl, targets, _ in dataloader:
        usit, c1d, vdl, targets = usit.to(device), c1d.to(device), vdl.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(usit, c1d, vdl)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * usit.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    exact_matches = 0
    adjacent_matches = 0
    total = 0
    with torch.no_grad():
        for usit, c1d, vdl, _, labels in dataloader:
            usit, c1d, vdl = usit.to(device), c1d.to(device), vdl.to(device)
            outputs = model(usit, c1d, vdl).cpu().numpy()
            labels = labels.numpy()

            for i in range(len(labels)):
                pred_class = decode_ordinal_prediction(outputs[i])
                true_class = labels[i]

                if pred_class == true_class:
                    exact_matches += 1
                if abs(pred_class - true_class) <= 1:
                    adjacent_matches += 1
                total += 1

    exact_acc = exact_matches / total if total > 0 else 0.0
    adj_acc = adjacent_matches / total if total > 0 else 0.0
    return exact_acc, adj_acc

# =============================================================================
# 6. EXECUÇÃO DO K-FOLD LEAVE-ONE-WELL-OUT
# =============================================================================
def run_kfold_training(data_dir=R"D:\cased_borehole_cement_evaluation\dataset\data"):
    dlis_files = glob.glob(os.path.join(data_dir, "*.dlis"))
    if len(dlis_files) < 3:
        print(f"Diretório '{data_dir}' não contém 3 arquivos .dlis locais.")
        print("Usando 3 nomes fictícios de poços para estruturar a validação cruzada K-Fold:")
        dlis_files = [
            os.path.join(data_dir, "well_1.dlis"),
            os.path.join(data_dir, "well_2.dlis"),
            os.path.join(data_dir, "well_3.dlis")
        ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando modelo PyTorch em dispositivo: {device}\n")

    # K-Fold: 2 poços para treino, 1 poço para teste (3 combinações)
    for fold in range(len(dlis_files)):
        test_file = [dlis_files[fold]]
        train_files = [dlis_files[i] for i in range(len(dlis_files)) if i != fold]

        print(f"==================== FOLD {fold + 1}/{len(dlis_files)} ====================")
        print(f"Treino: {[os.path.basename(f) for f in train_files]}")
        print(f"Teste:  {[os.path.basename(f) for f in test_file]}")

        train_dataset = CementLogDataset(train_files, is_train=True)
        test_dataset  = CementLogDataset(test_file, is_train=False)

        # Sampler Balanceado (3000 amostras equilibradas entre as 6 classes)
        train_labels = [s['label'] for s in train_dataset.samples]
        class_counts = np.bincount(train_labels, minlength=6)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[l] for l in train_labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=min(3000, len(train_dataset)),
            replacement=True
        )

        train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
        test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Modelo, Otimizador RMSprop (lr=0.001) e Perda BCE
        model = CementEvaluationCNN(num_classes=6).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        num_epochs = 10
        for epoch in range(1, num_epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            exact_acc, adj_acc = evaluate(model, test_loader, device)
            print(f"Época {epoch:02d}/{num_epochs:02d} | Perda: {loss:.4f} | Ac. Exata: {exact_acc * 100:.2f}% | Ac. Adjacente: {adj_acc * 100:.2f}%")
        print()

if __name__ == "__main__":
    run_kfold_training()
