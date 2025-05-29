import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
mpl.rcParams['axes.unicode_minus'] = False


def prepare_expression_data(max_gene=128) -> pd.DataFrame:
    """
    Load the dataset from a cct file and prepare it for analysis.
    """
    df = pd.read_csv("HS_CPTAC_GBM_rnaseq_fpkm_uq_log2.cct", sep="\t", index_col=0)
    df = df.transpose()  # shape: (num_samples, num_genes)

    # filter
    df = df.replace(0, pd.NA).dropna(axis=1)  # filter out genes with zero expression
    top_genes = df.var().sort_values(ascending=False).head(max_gene).index
    df = df[top_genes]  # keep only the top 5000 variant genes

    # normalize (z-score)
    df = (df - df.mean()) / df.std()

    # save
    np.save('HS_CPTAC_GBM_rnaseq_fpkm_uq_log2.npy', df.to_numpy())

    return df


# ======== VAE Model ========
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super(VAE, self).__init__()

        # Encoder
        encoder = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(0.2), ]
        for i in range(1, len(hidden_dims)):
            encoder.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            encoder.append(nn.ReLU())
            encoder.append(nn.Dropout(0.1))
        self.encoder = nn.Sequential(*encoder)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder = [nn.Linear(latent_dim, hidden_dims[-1]), nn.ReLU(), nn.Dropout(0.2), ]
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder.append(nn.Linear(hidden_dims[i], hidden_dims[i - 1]))
            decoder.append(nn.BatchNorm1d(hidden_dims[i - 1]))
            decoder.append(nn.ReLU())
            decoder.append(nn.Dropout(0.2))
        decoder.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ======== Loss Function with Split Output ========
def vae_loss(x_recon, x, mu, logvar, beta=1.0, return_parts=False):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_div
    if return_parts:
        return total_loss, recon_loss, kl_div
    else:
        return total_loss


# ======== Display loss changes ========
def plot_loss(train_loss, val_loss=None, train_recon_loss=None,
              val_recon_loss=None, train_kl_loss=None, val_kl_loss=None, lr=None):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    plt.plot(train_loss, label='Training Loss')
    if val_loss is not None:
        plt.plot(val_loss, label='Validation Loss')
    if train_recon_loss is not None:
        plt.plot(train_recon_loss, label='Training Recon Loss')
    if val_recon_loss is not None:
        plt.plot(val_recon_loss, label='Validation Recon Loss')
    if train_kl_loss is not None:
        plt.plot(train_kl_loss, label='Training KL Loss')
    if val_kl_loss is not None:
        plt.plot(val_kl_loss, label='Validation KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if lr is not None:
        plt.plot(lr, color='r', linestyle='-', label=f'Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ======== Train one epoch ========
def train_one_epoch(model, train_loader, optimizer, device='cpu', beta=1.0):
    model.train()
    total_loss = 0
    recon_loss = 0
    kl_loss = 0

    for batch in train_loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss, recon_loss_batch, kl_div = vae_loss(x_recon, x, mu, logvar, return_parts=True, beta=beta)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        recon_loss += recon_loss_batch.item()
        kl_loss += kl_div.item()

    n_train = len(train_loader.dataset)
    total_loss /= n_train
    recon_loss /= n_train
    kl_loss /= n_train

    return total_loss, recon_loss, kl_loss


# ======== Validation Function ========
def validate_vae(model, val_loader, device='cpu', beta=1.0):
    model.eval()
    val_total_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            x_recon, mu, logvar = model(x)
            loss, recon_loss, kl_div = vae_loss(x_recon, x, mu, logvar, return_parts=True, beta=beta)
            val_total_loss += loss.item()
            val_recon_loss += recon_loss.item()
            val_kl_loss += kl_div.item()

    n_val = len(val_loader.dataset)
    val_total_loss /= n_val
    val_recon_loss /= n_val
    val_kl_loss /= n_val

    return val_total_loss, val_recon_loss, val_kl_loss


# ======== Training Function with Detailed Loss Outputs ========
def train_vae(model, train_loader, val_loader=None, optimizer=None, scheduler=None, epochs=100, lr=1e-3,
              device='cuda', early_stop_patience=20, beta_max=4.0, beta_annealing=50):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)

    best_loss = float('inf')
    patience = 0
    best_model_state = None

    history_train_loss = []
    history_val_loss = []
    history_train_recon_loss = []
    history_val_recon_loss = []
    history_train_kl_loss = []
    history_val_kl_loss = []
    history_lr = []

    for epoch in range(epochs):
        train_total_loss, train_recon_loss, train_kl_loss = train_one_epoch(
            model, train_loader, optimizer, device, beta=min(beta_max, epoch / beta_annealing)
        )

        history_train_loss.append(train_total_loss)
        history_train_recon_loss.append(train_recon_loss)
        history_train_kl_loss.append(train_kl_loss)
        history_lr.append(optimizer.param_groups[0]['lr'])

        if not val_loader:
            print(f"Epoch {epoch + 1:3d} | "
                  f"Train: total {train_total_loss:.2f}, recon {train_recon_loss:.2f}, KL {train_kl_loss:.2f}")
            return

        val_total_loss, val_recon_loss, val_kl_loss = validate_vae(model, val_loader, device)

        history_val_loss.append(val_total_loss)
        history_val_recon_loss.append(val_recon_loss)
        history_val_kl_loss.append(val_kl_loss)

        print(f"Epoch {epoch + 1:3d} | "
              f"Train: total {train_total_loss:.2f}, recon {train_recon_loss:.2f}, KL {train_kl_loss:.2f} | "
              f"Val: total {val_total_loss:.2f}, recon {val_recon_loss:.2f}, KL {val_kl_loss:.2f}")

        if scheduler is not None:
            scheduler.step(val_total_loss)

        # Early stopping
        if val_total_loss < best_loss:
            best_loss = val_total_loss
            patience = 0
            best_model_state = model.state_dict()
        else:
            patience += 1
            if patience >= early_stop_patience:
                print("Early stopping triggered.")
                break

    plot_loss(history_train_loss, history_val_loss,
              history_train_recon_loss, history_val_recon_loss,
              history_train_kl_loss, history_val_kl_loss, history_lr)

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


# ======== Latent Space Visualization ========
def visualize_latent_space_tsne(model, data_loader, device, title, classes=None, use_sampled_z=False, sample_names=None,
                                clinical_data=None, color_by=None):
    """
    使用t-SNE視覺化VAE的latent space。可以選擇使用mu或z（mu + sigma * epsilon）
    新增功能：
    1. 顯示每個點對應的樣本名稱
    2. 支援使用臨床資料中的特徵進行顏色標記

    Args:
        model: VAE模型
        data_loader: 資料載入器
        device: 運算裝置
        title: 圖形標題
        classes: 類別標籤
        use_sampled_z: 是否使用採樣的z（否則使用mu）
        sample_names: 樣本名稱列表
        clinical_data: 包含臨床資料的DataFrame
        color_by: 要使用哪個臨床特徵進行顏色標記（例如 'gender', 'tumor_site_curated'）
    """
    model.eval()
    latent_vectors = []
    class_labels = []
    sample_indices = []  # 記錄每個batch中的樣本索引
    current_idx = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            batch_size = x.size(0)

            if hasattr(model, 'encode'):
                mu, logvar = model.encode(x)
            else:
                raise ValueError("模型需包含 encode(x) → mu, logvar 方法")

            if use_sampled_z:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                latent_vectors.append(z.cpu().numpy())
            else:
                latent_vectors.append(mu.cpu().numpy())

            # 記錄樣本索引
            sample_indices.extend(list(range(current_idx, current_idx + batch_size)))
            current_idx += batch_size

            # 提取 class label（如果存在）
            if len(batch) > 1 and classes is not None:
                class_labels.append(batch[1].cpu().numpy())

    latent_vectors = np.vstack(latent_vectors)
    if class_labels and classes is not None:
        class_labels = np.concatenate(class_labels)
    else:
        class_labels = None

    # 1. t-SNE 可視化
    print("Running t-SNE...")
    z_tsne = TSNE(n_components=2, perplexity=8, random_state=42).fit_transform(latent_vectors)

    plt.figure(figsize=(14, 12))

    # 根據臨床資料進行顏色標記
    if clinical_data is not None and color_by is not None and sample_names is not None:
        # 獲取當前批次中樣本的臨床資料
        point_labels = [sample_names[i] for i in sample_indices]

        # 從臨床資料中提取要使用的特徵
        color_values = []
        legend_labels = []
        valid_indices = []  # 追蹤有效的樣本索引

        for i, sample_id in enumerate(point_labels):
            if sample_id in clinical_data.index:
                color_value = clinical_data.loc[sample_id, color_by]
                # 處理可能的NA值
                if pd.isna(color_value):
                    color_value = "NA"
                color_values.append(color_value)
                if color_value not in legend_labels:
                    legend_labels.append(color_value)
                valid_indices.append(i)
            else:
                # 如果樣本ID不在臨床資料中，跳過該樣本或標記為"Unknown"
                continue

        # 將類別特徵轉換為數值
        if not all(
                isinstance(v, (int, float)) or
                (isinstance(v, str) and v.replace('.', '', 1).isdigit())
                for v in color_values):
            # 分類特徵
            unique_values = list(set(color_values))
            color_dict = {value: i for i, value in enumerate(unique_values)}
            color_nums = [color_dict[value] for value in color_values]

            # 使用適合分類資料的顏色映射
            cmap = plt.cm.get_cmap('tab20', len(unique_values))

            # 只保留有效的樣本點
            valid_z_tsne = z_tsne[valid_indices]

            # 繪製散點圖
            scatter = plt.scatter(valid_z_tsne[:, 0], valid_z_tsne[:, 1], c=color_nums, cmap=cmap, alpha=0.8, s=80)

            # 創建圖例
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=cmap(color_dict[value]),
                                          markersize=10, label=str(value))
                               for value in unique_values]
            plt.legend(handles=legend_elements, title=color_by,
                       loc='upper right', bbox_to_anchor=(1.15, 1))
        else:
            # 數值特徵
            valid_z_tsne = z_tsne[valid_indices]
            color_nums = [float(v) for v in color_values]

            # 使用適合連續資料的顏色映射
            scatter = plt.scatter(valid_z_tsne[:, 0], valid_z_tsne[:, 1],
                                  c=color_nums, cmap='viridis', alpha=0.8, s=80)
            cbar = plt.colorbar(scatter)
            cbar.set_label(color_by)

        # 為每個點添加樣本名稱標籤
        for i, (x, y) in enumerate(valid_z_tsne):
            sample_id = point_labels[valid_indices[i]]
            plt.annotate(sample_id, (x, y), fontsize=8, alpha=0.7)

    else:
        # 如果沒有提供臨床資料或顏色標記特徵，使用原始方法
        if class_labels is not None:
            scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=class_labels, cmap='tab10', alpha=0.7, s=80)
            plt.colorbar(scatter, label='Class Label')
        else:
            scatter = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], alpha=0.7, color='gray', s=80)

        # 標示樣本名稱
        if sample_names is not None:
            point_labels = [sample_names[i] for i in sample_indices]
            for i, (x, y) in enumerate(z_tsne):
                plt.annotate(point_labels[i], (x, y), fontsize=8, alpha=0.7)

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. 每個 latent 維度的 histogram
    print("Plotting latent dimension activations...")
    dim_to_plot = min(8, latent_vectors.shape[1])
    fig, axes = plt.subplots(1, dim_to_plot, figsize=(20, 3))
    for i in range(dim_to_plot):
        sns.histplot(latent_vectors[:, i], kde=True, ax=axes[i])
        axes[i].set_title(f'z_{i}')
    plt.suptitle("每個 latent 維度的分布")
    plt.tight_layout()
    plt.show()

    return latent_vectors, z_tsne, sample_indices


# ======== Gene Reconstruction Error ========
def compute_gene_reconstruction_errors(model, data_tensor, device='cuda'):
    model.eval()
    data_tensor = data_tensor.to(device)
    with torch.no_grad():
        x_recon, _, _ = model(data_tensor)
        errors = (x_recon - data_tensor) ** 2  # shape: (n_samples, n_genes)
        gene_errors = errors.mean(dim=0).cpu().numpy()  # shape: (n_genes,)
    return gene_errors


def get_top_genes_by_error(gene_errors, gene_names, top_n=5):
    error_df = pd.DataFrame({'gene': gene_names, 'recon_error': gene_errors})
    error_df_sorted = error_df.sort_values(by='recon_error', ascending=False)
    return error_df_sorted.head(top_n)


def plot_top_gene_errors(error_df_top):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='recon_error', y='gene', data=error_df_top, palette='coolwarm')
    plt.title('Top Genes by Reconstruction Error')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Gene')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_latent_space_pca(model, data_loader, device, title, classes=None, use_sampled_z=False, sample_names=None,
                               clinical_data=None, color_by=None):
    """
    Visualize the latent space of a VAE using PCA.
    Args:
        model: VAE model
        data_loader: DataLoader for the dataset
        device: Device to run the model on
        title: Title of the plot
        classes: Class labels (optional)
        use_sampled_z: Whether to use sampled z (mu + sigma * epsilon) or just mu
        sample_names: List of sample names
        clinical_data: Clinical data DataFrame
        color_by: Clinical feature to color the points by
    """
    model.eval()
    latent_vectors = []
    class_labels = []
    sample_indices = []
    current_idx = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            batch_size = x.size(0)

            if hasattr(model, 'encode'):
                mu, logvar = model.encode(x)
            else:
                raise ValueError("Model must have an encode(x) → mu, logvar method")

            if use_sampled_z:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                latent_vectors.append(z.cpu().numpy())
            else:
                latent_vectors.append(mu.cpu().numpy())

            sample_indices.extend(list(range(current_idx, current_idx + batch_size)))
            current_idx += batch_size

            if len(batch) > 1 and classes is not None:
                class_labels.append(batch[1].cpu().numpy())

    latent_vectors = np.vstack(latent_vectors)
    if class_labels and classes is not None:
        class_labels = np.concatenate(class_labels)
    else:
        class_labels = None

    # PCA Transformation
    print("Running PCA...")
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(14, 12))

    # Coloring by clinical data
    if clinical_data is not None and color_by is not None and sample_names is not None:
        point_labels = [sample_names[i] for i in sample_indices]

        color_values = []
        legend_labels = []
        valid_indices = []

        for i, sample_id in enumerate(point_labels):
            if sample_id in clinical_data.index:
                color_value = clinical_data.loc[sample_id, color_by]
                if pd.isna(color_value):
                    color_value = "NA"
                color_values.append(color_value)
                if color_value not in legend_labels:
                    legend_labels.append(color_value)
                valid_indices.append(i)
            else:
                continue

        if not all(isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit())
                   for v in color_values):
            unique_values = list(set(color_values))
            color_dict = {value: i for i, value in enumerate(unique_values)}
            color_nums = [color_dict[value] for value in color_values]

            cmap = plt.cm.get_cmap('tab20', len(unique_values))
            valid_z_pca = z_pca[valid_indices]

            scatter = plt.scatter(valid_z_pca[:, 0], valid_z_pca[:, 1], c=color_nums, cmap=cmap, alpha=0.8, s=80)

            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=cmap(color_dict[value]),
                                          markersize=10, label=str(value))
                               for value in unique_values]
            plt.legend(handles=legend_elements, title=color_by,
                       loc='upper right', bbox_to_anchor=(1.15, 1))
        else:
            valid_z_pca = z_pca[valid_indices]
            color_nums = [float(v) for v in color_values]

            scatter = plt.scatter(valid_z_pca[:, 0], valid_z_pca[:, 1],
                                  c=color_nums, cmap='viridis', alpha=0.8, s=80)
            cbar = plt.colorbar(scatter)
            cbar.set_label(color_by)

        for i, (x, y) in enumerate(valid_z_pca):
            sample_id = point_labels[valid_indices[i]]
            plt.annotate(sample_id, (x, y), fontsize=8, alpha=0.7)

    else:
        if class_labels is not None:
            scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=class_labels, cmap='tab10', alpha=0.7, s=80)
            plt.colorbar(scatter, label='Class Label')
        else:
            scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.7, color='gray', s=80)

        if sample_names is not None:
            point_labels = [sample_names[i] for i in sample_indices]
            for i, (x, y) in enumerate(z_pca):
                plt.annotate(point_labels[i], (x, y), fontsize=8, alpha=0.7)

    plt.title(title)
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ======== Example Usage ========
def run():
    # Load data
    X = np.load("HS_CPTAC_GBM_rnaseq_fpkm_uq_log2.npy")  # (num_samples, num_genes)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # 從原始檔案讀取樣本名稱
    # 假設RNA-seq原始資料是由sample為column, gene為row組成
    try:
        # 嘗試讀取包含樣本名稱的原始檔案
        original_data = pd.read_csv("HS_CPTAC_GBM_rnaseq_fpkm_uq_log2.cct", sep="\t", index_col=0)
        # 獲取樣本名稱（列名）
        sample_names = original_data.columns.tolist()
        print(f"成功載入 {len(sample_names)} 個樣本名稱")
    except Exception as e:
        print(f"無法載入樣本名稱: {e}")
        # 如果無法載入，使用序號作為樣本名稱
        sample_names = [f"Sample_{i}" for i in range(X.shape[0])]
        print(f"使用預設樣本名稱: Sample_0 to Sample_{X.shape[0] - 1}")

    # 讀取臨床資料
    try:
        clinical_data = pd.read_csv("HS_CPTAC_GBM_CLI.tsi", sep="\t", index_col=0)
        print(f"成功載入臨床資料，包含 {clinical_data.shape[0]} 筆記錄，{clinical_data.shape[1]} 個特徵")
        print("可用的臨床特徵: ", clinical_data.columns.tolist())

        # 檢查臨床資料與樣本名稱的對應關係
        matching_samples = set(sample_names).intersection(set(clinical_data.index))
        print(f"{len(matching_samples)} 個樣本在臨床資料中有對應的記錄")
    except Exception as e:
        print(f"無法載入臨床資料: {e}")
        clinical_data = None

    # Train/Val split
    # 注意：這裡我們需要追蹤哪些樣本被分到訓練集和驗證集
    train_indices, val_indices = train_test_split(
        np.arange(len(X_tensor)),
        test_size=0.2,
        random_state=42
    )

    train_data = X_tensor[train_indices]
    val_data = X_tensor[val_indices]

    # 記錄訓練集和驗證集對應的樣本名稱
    train_sample_names = [sample_names[i] for i in train_indices]
    val_sample_names = [sample_names[i] for i in val_indices]

    train_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=32)

    # Model init
    input_dim = X.shape[1]
    model = VAE(input_dim=input_dim, latent_dim=8, hidden_dims=[64, 32])

    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)
    trained_model = train_vae(model, train_loader, val_loader, optimizer=optimizer, scheduler=scheduler, epochs=300,
                              lr=1e-3, device=device, early_stop_patience=500000, beta_max=2, beta_annealing=50)

    # Visualize latent space with sample names and clinical features
    # 注意：由於DataLoader可能會打亂訓練數據的順序，所以我們創建一個不打亂順序的loader用於視覺化
    train_viz_loader = DataLoader(TensorDataset(train_data), batch_size=32, shuffle=False)
    val_viz_loader = DataLoader(TensorDataset(val_data), batch_size=32, shuffle=False)

    # 創建一個函數，允許使用者選擇要使用哪個臨床特徵進行顏色標記
    def visualize_with_clinical_feature(feature_name=None):
        if feature_name is None:
            return

        # 視覺化訓練數據的latent space，並使用臨床特徵進行顏色標記
        visualize_latent_space_tsne(
            trained_model, train_viz_loader, device, f'Training Data Latent Space (colored by {feature_name})',
            use_sampled_z=True, sample_names=train_sample_names,
            clinical_data=clinical_data, color_by=feature_name
        )

        # 視覺化驗證數據的latent space，並使用臨床特徵進行顏色標記
        visualize_latent_space_tsne(
            trained_model, val_viz_loader, device, f'Validation Data Latent Space (colored by {feature_name})',
            use_sampled_z=True, sample_names=val_sample_names,
            clinical_data=clinical_data, color_by=feature_name
        )

    # 依據臨床資料可用性，顯示各種特徵的視覺化
    if clinical_data is not None:
        visualize_with_clinical_feature('bmi')
        visualize_latent_space_pca(
            trained_model, train_viz_loader, device, 'Latent Space PCA (z) for training data',
            use_sampled_z=True, sample_names=train_sample_names,
            clinical_data=clinical_data, color_by='bmi'
        )
        visualize_latent_space_pca(
            trained_model, train_viz_loader, device, 'Latent Space PCA (z) for training data',
            use_sampled_z=True, sample_names=train_sample_names,
            clinical_data=clinical_data, color_by='age'
        )
    else:
        visualize_latent_space_tsne(
            trained_model, train_viz_loader, device, 'Latent Space t-SNE (z) for training data',
            use_sampled_z=True, sample_names=train_sample_names
        )

        visualize_latent_space_tsne(
            trained_model, val_viz_loader, device, 'Latent Space t-SNE (z) for validation data',
            use_sampled_z=True, sample_names=val_sample_names
        )

    # Gene-level Reconstruction Error Analysis
    gene_names = pd.read_csv("HS_CPTAC_GBM_rnaseq_fpkm_uq_log2.cct", sep="\t", index_col=0).transpose()
    gene_names = gene_names.replace(0, pd.NA).dropna(axis=1)
    top_genes = gene_names.var().sort_values(ascending=False).head(128).index.tolist()

    all_data = torch.tensor(np.load("HS_CPTAC_GBM_rnaseq_fpkm_uq_log2.npy"), dtype=torch.float32)
    gene_errors = compute_gene_reconstruction_errors(trained_model, all_data, device)
    top_error_df = get_top_genes_by_error(gene_errors, gene_names=top_genes, top_n=20)
    print("Top genes with highest reconstruction error:")
    print(top_error_df)

    plot_top_gene_errors(top_error_df)

    # ========= 潛在空間與臨床特徵的關聯分析 =========
    print("\n分析潛在變數與臨床特徵之間的關聯...\n")
    trained_model.eval()
    with torch.no_grad():
        full_mu, _ = trained_model.encode(X_tensor.to(device))
        latent_array = full_mu.cpu().numpy()
        latent_df = pd.DataFrame(latent_array, index=sample_names,
                                 columns=[f"z{i}" for i in range(latent_array.shape[1])])

    # 合併臨床資料
    matched_clinical = clinical_data.loc[latent_df.index.intersection(clinical_data.index)]
    latent_df = latent_df.loc[matched_clinical.index]
    full_df = pd.concat([latent_df, matched_clinical], axis=1)

    # 分析與連續變數（例如 age, bmi）的Spearman相關性
    from scipy.stats import spearmanr

    for feature in ['age', 'bmi']:
        if feature in full_df.columns:
            print(f"\n[Spearman correlation] 潛在變數 vs {feature}:")
            for z in latent_df.columns:
                corr, pval = spearmanr(full_df[z], full_df[feature], nan_policy='omit')
                print(f"{z}: rho = {corr:.3f}, p = {pval:.4f}")

    # 分析與分類變數（例如 gender, ethnicity_self_identify）的分布差異
    import seaborn as sns
    import matplotlib.pyplot as plt

    for cat_feature in ['gender', 'ethnicity_self_identify', 'smoking_history', 'tumor_site_curated']:
        if cat_feature in full_df.columns:
            print(f"\n[Boxplot] 潛在變數 vs {cat_feature}")
            for z in latent_df.columns:
                plt.figure(figsize=(6, 4))
                sns.boxplot(data=full_df, x=cat_feature, y=z)
                plt.title(f"{z} vs {cat_feature}")
                plt.xticks(rotation=30)
                plt.tight_layout()
                plt.show()


if __name__ == "__main__":
    # print(prepare_expression_data())
    prepare_expression_data(max_gene=128)
    run()
