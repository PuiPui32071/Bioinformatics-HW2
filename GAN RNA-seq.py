import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from tqdm import tqdm
import os
from typing import List


# 設置隨機種子以確保結果可以復現
torch.manual_seed(42)
np.random.seed(42)


def preprocess_data(max_gene=5000):
    """
    Load the dataset from a cct file and prepare it for analysis.
    """
    df = pd.read_csv("HS_CPTAC_GBM_rnaseq_fpkm_uq_log2.cct", sep="\t", index_col=0)
    df = df.transpose()  # shape: (num_samples, num_genes)

    # filter
    df = df.replace(0, pd.NA).dropna(axis=1)  # filter out genes with zero expression
    top_genes = df.var().sort_values(ascending=False).head(max_gene).index
    df = df[top_genes]  # keep only the top variant genes

    df.transpose().to_csv(f"top_{max_gene}_genes.csv", sep="\t")

    # 標準化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)

    return data_scaled, top_genes, scaler


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims: List[int]):
        """
        初始化生成器網絡

        參數:
            latent_dim: 潛在空間維度
            output_dim: 輸出維度 (基因數量)
            hidden_dims: 隱藏層維度列表
        """
        super(Generator, self).__init__()

        # 構建網絡結構
        layers = [
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        ]

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims: List[int]):
        """
        初始化判別器網絡

        參數:
            input_dim: 輸入維度 (基因數量)
            hidden_dims: 隱藏層維度列表
        """
        super(Discriminator, self).__init__()

        # 構建網絡結構
        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            # nn.Dropout(0.2),
        ]

        # 第一層從基因表達空間到第一個隱藏層

        # 添加中間隱藏層
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.2))

        # 最後一層輸出到單一判別值
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class WGAN_GP(nn.Module):
    def __init__(self, gene_dim, latent_dim=128, hidden_dims_g=[512, 1024, 2048],
                 hidden_dims_d=[2048, 1024, 512], critic_iterations=5, lambda_gp=10):
        """
        初始化WGAN-GP模型

        參數:
            gene_dim: 基因數量
            latent_dim: 潛在空間維度
            hidden_dims_g: 生成器隱藏層維度列表
            hidden_dims_d: 判別器隱藏層維度列表
            critic_iterations: 每訓練一次生成器，判別器訓練的次數
            lambda_gp: 梯度懲罰的權重
        """
        super(WGAN_GP, self).__init__()

        self.gene_dim = gene_dim
        self.latent_dim = latent_dim
        self.critic_iterations = critic_iterations
        self.lambda_gp = lambda_gp

        # 初始化生成器和判別器
        self.generator = Generator(latent_dim, gene_dim, hidden_dims_g)
        self.discriminator = Discriminator(gene_dim, hidden_dims_d)

    def compute_gradient_penalty(self, real_samples, fake_samples, device):
        """
        計算梯度懲罰

        參數:
            real_samples: 真實數據
            fake_samples: 生成數據
            device: 計算設備

        返回:
            計算得到的梯度懲罰
        """
        # 隨機生成插值係數
        alpha = torch.rand(real_samples.size(0), 1).to(device)
        alpha = alpha.expand_as(real_samples)

        # 生成插值樣本
        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
        interpolates = interpolates.requires_grad_(True)

        # 計算判別器對插值樣本的輸出
        d_interpolates = self.discriminator(interpolates)

        # 創建全1的張量
        fake = torch.ones(d_interpolates.size()).to(device)

        # 計算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # 計算梯度的L2範數
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def generate_samples(self, num_samples):
        """
        生成樣本

        參數:
            num_samples: 要生成的樣本數量

        返回:
            生成的樣本張量
        """
        # 生成潛在空間噪聲
        z = torch.randn(num_samples, self.latent_dim).to(next(self.generator.parameters()).device)

        # 生成樣本
        with torch.no_grad():
            samples = self.generator(z)

        return samples

    def train_model(self, dataloader, num_epochs=200, batch_size=32, lr_d=5e-5, lr_g=5e-5, betas=(0.5, 0.9),
                    save_interval=10, save_dir='./models', device='cuda'):
        """
        訓練WGAN模型

        參數:
            dataloader: 包含訓練數據的DataLoader
            num_epochs: 訓練輪數
            batch_size: 批量大小
            lr_d: Discriminator學習率
            lr_g: Generator學習率
            betas: Adam優化器的beta參數
            save_interval: 每多少輪儲存一次模型
            save_dir: 模型儲存目錄
            device: 訓練設備
        """
        # 移動模型到指定設備
        self.generator.to(device)
        self.discriminator.to(device)

        # 創建優化器
        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=betas)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=betas)

        # 創建儲存目錄
        os.makedirs(save_dir, exist_ok=True)

        # 紀錄訓練過程
        g_losses = []
        d_losses = []

        # 訓練循環
        for epoch in range(num_epochs):
            running_g_loss = 0.0
            running_d_loss = 0.0

            for i, real_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):

                # 準備真實數據
                real_data = real_data[0].to(device)  # 取出數據部分
                batch_size = real_data.size(0)

                # ---------------------
                # 訓練判別器
                # ---------------------
                for _ in range(self.critic_iterations):
                    # 清除梯度
                    optimizer_d.zero_grad()

                    # 生成假樣本
                    z = torch.randn(batch_size, self.latent_dim).to(device)
                    fake_data = self.generator(z)

                    # 計算判別器輸出
                    real_output = self.discriminator(real_data).view(-1)
                    fake_output = self.discriminator(fake_data.detach()).view(-1)

                    # 計算梯度懲罰
                    gradient_penalty = self.compute_gradient_penalty(real_data, fake_data.detach(), device)

                    # 計算WGAN-GP判別器損失
                    d_loss = torch.mean(fake_output) - torch.mean(real_output) + self.lambda_gp * gradient_penalty

                    # 反向傳播
                    d_loss.backward()
                    optimizer_d.step()

                # ---------------------
                # 訓練生成器
                # ---------------------
                optimizer_g.zero_grad()

                # 生成假樣本
                z = torch.randn(batch_size, self.latent_dim).to(device)
                fake_data = self.generator(z)

                # 計算生成器損失
                fake_output = self.discriminator(fake_data).view(-1)
                g_loss = -torch.mean(fake_output)

                # 反向傳播
                g_loss.backward()
                optimizer_g.step()

                # 紀錄損失
                running_d_loss += d_loss.item()
                running_g_loss += g_loss.item()

            # 計算平均損失
            avg_d_loss = running_d_loss / len(dataloader)
            avg_g_loss = running_g_loss / len(dataloader)

            # 紀錄損失
            d_losses.append(avg_d_loss)
            g_losses.append(avg_g_loss)

            # 打印訓練進度
            print(f"Epoch [{epoch + 1}/{num_epochs}] d_loss: {avg_d_loss:.4f}, g_loss: {avg_g_loss:.4f}")

            # 定期儲存模型
            if (epoch + 1) % save_interval == 0:
                torch.save({
                    'generator_state_dict': self.generator.state_dict(),
                    'discriminator_state_dict': self.discriminator.state_dict(),
                    'optimizer_g_state_dict': optimizer_g.state_dict(),
                    'optimizer_d_state_dict': optimizer_d.state_dict(),
                    'epoch': epoch,
                }, f"{save_dir}/wgan_epoch_{epoch + 1}.pt")

        # 繪製損失曲線
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label='Generator Loss')
        plt.plot(d_losses, label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/loss_curve.png")
        plt.close()

        # 儲存最終模型
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
        }, f"{save_dir}/wgan_final.pt")

        return g_losses, d_losses


def visualize_data(real_data, fake_data, save_path='./results'):
    """
    可視化真實數據和生成數據

    參數:
        real_data: 真實數據
        fake_data: 生成數據
        save_path: 結果儲存路徑
    """
    # 創建儲存目錄
    os.makedirs(save_path, exist_ok=True)

    # 合併數據
    combined_data = np.vstack([real_data, fake_data])
    labels = np.array(['Real'] * len(real_data) + ['Generated'] * len(fake_data))

    # PCA降維
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_data)

    # 繪製PCA圖
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, alpha=0.7)
    plt.title('PCA Visualization')
    plt.savefig(f"{save_path}/pca_visualization.png")
    plt.close()

    # t-SNE降維
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(combined_data)

    # 繪製t-SNE圖
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, alpha=0.7)
    plt.title('t-SNE Visualization')
    plt.savefig(f"{save_path}/tsne_visualization.png")
    plt.close()

    # 計算並繪製基因表達分布
    plt.figure(figsize=(12, 6))
    plt.hist(real_data.flatten(), bins=50, alpha=0.5, label='Real')
    plt.hist(fake_data.flatten(), bins=50, alpha=0.5, label='Generated')
    plt.legend()
    plt.title('Gene Expression Distribution')
    plt.savefig(f"{save_path}/expression_distribution.png")
    plt.close()

    # 計算並繪製相關性熱圖
    real_corr = np.corrcoef(real_data.T)
    fake_corr = np.corrcoef(fake_data.T)

    plt.figure(figsize=(10, 8))
    sns.heatmap(real_corr[:50, :50], cmap='coolwarm')
    plt.title('Real Data Correlation (Top 50 genes)')
    plt.savefig(f"{save_path}/real_correlation.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(fake_corr[:50, :50], cmap='coolwarm')
    plt.title('Generated Data Correlation (Top 50 genes)')
    plt.savefig(f"{save_path}/fake_correlation.png")
    plt.close()


def compute_mmd(x, y, kernel='rbf', gamma=1.0):
    """
    計算 Maximum Mean Discrepancy (MMD)

    參數:
        x (ndarray): 真實資料，形狀為 (n_samples, n_features)
        y (ndarray): 生成資料，形狀為 (n_samples, n_features)
        kernel (str): 使用的核函數 ('rbf' 或 'linear')
        gamma (float): RBF 核的 gamma 參數

    返回:
        mmd值 (float)
    """
    from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

    if kernel == 'rbf':
        xx = rbf_kernel(x, x, gamma=gamma)
        yy = rbf_kernel(y, y, gamma=gamma)
        xy = rbf_kernel(x, y, gamma=gamma)
    elif kernel == 'linear':
        xx = linear_kernel(x, x)
        yy = linear_kernel(y, y)
        xy = linear_kernel(x, y)
    else:
        raise ValueError("Unsupported kernel")

    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd


def run_glioblastoma_gan(hidden_dims_g, hidden_dims_d,
                         top_n_genes=2000, latent_dim=128, batch_size=32,
                         num_epochs=100, save_interval=10, save_dir='./models', result_dir='./results'):
    """
    運行GAN模型訓練和評估

    參數:
        data_path: 數據文件路徑
        top_n_genes: 選擇變異最大的前N個基因
        latent_dim: 潛在空間維度
        batch_size: 批量大小
        num_epochs: 訓練輪數
        save_interval: 每多少輪儲存一次模型
        save_dir: 模型儲存路徑
        result_dir: 結果儲存路徑
    """
    # 確認是否有GPU可用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 預處理數據
    data_scaled, gene_names, scaler = preprocess_data(top_n_genes)

    # 轉換為PyTorch張量
    data_tensor = torch.FloatTensor(data_scaled)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    wgan = WGAN_GP(gene_dim=top_n_genes, latent_dim=latent_dim,
                   hidden_dims_d=hidden_dims_d, hidden_dims_g=hidden_dims_g,
                   critic_iterations=5)

    # 訓練模型
    g_losses, d_losses = wgan.train_model(
        dataloader,
        num_epochs=num_epochs,
        batch_size=batch_size,
        save_interval=save_interval,
        save_dir=save_dir,
        device=device,
        lr_g=1e-4,
        lr_d=1e-4,
    )

    # 生成樣本並逆轉標準化
    wgan.generator.eval()
    wgan.generator.to(device)
    generated_samples = wgan.generate_samples(len(data_scaled)).cpu().numpy()

    # 可視化比較
    visualize_data(data_scaled, generated_samples, save_path=result_dir)

    # 計算 MMD 分數
    mmd_score = compute_mmd(data_scaled, generated_samples, kernel='rbf', gamma=1.0 / data_scaled.shape[1])
    print(f"MMD between real and generated data: {mmd_score:.4f}")

    # 保存生成樣本
    generated_df = pd.DataFrame(generated_samples, columns=gene_names)
    generated_df = scaler.inverse_transform(generated_df)
    pd.DataFrame(generated_df, columns=gene_names).transpose().to_csv(f"{result_dir}/generated_samples.csv")

    print("Training and evaluation complete!")


if __name__ == "__main__":
    # 主程式入口點
    # 使用示例：
    run_glioblastoma_gan([32, 48, 64], [32, 64],
                         top_n_genes=128, latent_dim=16,
                         batch_size=32, num_epochs=2000, save_interval=10,
                         save_dir='./models', result_dir='./results')
    # real = pd.read_csv("top_128_genes.csv", sep="\t", index_col=0)
    # real = real.transpose().to_numpy()
    # generated = pd.read_csv("results/generated_samples.csv", index_col=0)
    # generated = generated.transpose().to_numpy()
    #
    # visualize_data(real, generated, save_path='./results_unz')
