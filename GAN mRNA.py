import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter

from tqdm import tqdm


# 設定隨機種子以確保結果可重現
def set_seed(seed: int = 42) -> None:
    """設定隨機種子以確保結果可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)

# 配置參數
SPECIES_INTERESTED = {'hsa': 'Human', 'mmu': 'Mouse'}
TARGET_LENGTH = 22  # mRNA序列的目標長度
NUCLEOTIDES = ['A', 'C', 'G', 'U']  # RNA核甘酸
BATCH_SIZE = 64
LATENT_DIM = 100  # 生成器潛在空間的維度
EPOCHS = 1500
N_CRITIC = 5  # 每訓練一次生成器，判別器訓練的次數
LAMBDA_GP = 10  # 梯度懲罰係數
LEARNING_RATE = 1e-5

# 判斷使用CPU還是GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")


# 數據處理函數
def process_fasta_file(file_path: str, target_length: int = TARGET_LENGTH,
                       species_interested=None) -> Tuple[List[str], List[str]]:
    """處理FASTA文件，過濾合法序列並標準化長度"""
    if species_interested is None:
        species_interested = SPECIES_INTERESTED

    sequences = []
    species = []

    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq).upper()
        species_name = record.description.split()[0].split('-')[0] if len(record.description.split()) > 1 else "Unknown"
        if species_name not in species_interested:
            continue
        # if record.description.split()[0].split('-')[-1] != '3p':
        #     continue

        # 過濾只包含合法核甘酸的序列
        if all(nucleotide in NUCLEOTIDES for nucleotide in seq.replace('T', 'U')):
            # 將T轉換為U（RNA格式）
            seq = seq.replace('T', 'U')
            species_name = species_interested[species_name]

            # 標準化長度
            if len(seq) == target_length:
                sequences.append(seq)
                species.append(species_name)
            elif len(seq) < target_length:
                # 對於較短的序列，用隨機核甘酸填充到目標長度
                padding = ''.join(random.choices(NUCLEOTIDES, k=target_length - len(seq)))
                sequences.append(seq + padding)
                species.append(species_name)
            elif len(seq) > target_length:
                # 對於較長的序列，截斷到目標長度
                sequences.append(seq[:target_length])
                species.append(species_name)

    return sequences, species


# 將序列轉換為one-hot編碼
def sequence_to_onehot(sequence: str) -> np.ndarray:
    """將核甘酸序列轉換為one-hot編碼"""
    # 建立核甘酸到索引的映射
    nucleotide_to_idx = {n: i for i, n in enumerate(NUCLEOTIDES)}

    # 創建one-hot編碼
    onehot = np.zeros((len(sequence), len(NUCLEOTIDES)), dtype=np.float32)
    for i, nucleotide in enumerate(sequence):
        if nucleotide in nucleotide_to_idx:
            onehot[i, nucleotide_to_idx[nucleotide]] = 1

    return onehot


# 從one-hot編碼轉回序列
def onehot_to_sequence(onehot: np.ndarray) -> str:
    """將one-hot編碼轉換回核甘酸序列"""
    # 取得每個位置概率最高的核甘酸
    indices = np.argmax(onehot, axis=1)

    # 將索引轉換為核甘酸
    sequence = ''.join([NUCLEOTIDES[idx] for idx in indices])

    return sequence


# 創建PyTorch資料集
class MRNADataset(Dataset):
    """mRNA序列資料集"""

    def __init__(self, sequences: List[str]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # 轉換為one-hot編碼並轉為Tensor
        return torch.from_numpy(sequence_to_onehot(self.sequences[idx]))


# 創建生成器模型(使用CNN)
class Generator(nn.Module):
    """生成器模型"""

    def __init__(self, latent_dim: int, seq_length: int, num_nucleotides: int):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.num_nucleotides = num_nucleotides

        # 計算中間層大小
        self.hidden_dim = 128

        # 全連接層將潛在向量轉換為適當的大小
        self.fc = nn.Linear(latent_dim, self.hidden_dim * seq_length)

        # 卷積轉置層
        self.conv_layers = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(self.hidden_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(32, num_nucleotides, kernel_size=5, padding=2)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z的形狀: [batch_size, latent_dim]
        x = self.fc(z)
        # 重新形狀為 [batch_size, hidden_dim, seq_length]
        x = x.view(-1, self.hidden_dim, self.seq_length)
        # 通過卷積層
        x = self.conv_layers(x)
        # 轉置為 [batch_size, seq_length, num_nucleotides]
        x = x.permute(0, 2, 1)
        # 使用softmax確保每個位置的核甘酸概率和為1
        return torch.softmax(x, dim=2)


# 創建判別器模型(使用CNN)
class Discriminator(nn.Module):
    """判別器模型"""

    def __init__(self, seq_length: int, num_nucleotides: int):
        super(Discriminator, self).__init__()

        # 卷積層
        self.conv_layers = nn.Sequential(
            # 輸入: [batch_size, num_nucleotides, seq_length]
            nn.Conv1d(num_nucleotides, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3)
        )

        # 計算卷積層輸出大小
        conv_output_size = 128 * seq_length

        # 全連接層
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x的形狀: [batch_size, seq_length, num_nucleotides]
        # 轉置為 [batch_size, num_nucleotides, seq_length]
        x = x.permute(0, 2, 1)
        # 通過卷積層
        x = self.conv_layers(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 通過全連接層
        return self.fc_layers(x)


# 計算Wasserstein損失
def wasserstein_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """計算Wasserstein損失"""
    return torch.mean(y_true * y_pred)


# 計算梯度懲罰
def gradient_penalty(discriminator: nn.Module, real_samples: torch.Tensor, fake_samples: torch.Tensor,
                     device: torch.device) -> torch.Tensor:
    """計算梯度懲罰"""
    # 隨機權重項
    alpha = torch.rand((real_samples.size(0), 1, 1), device=device)
    # 在真實和生成的樣本之間創建插值
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # 判別器對插值的輸出
    d_interpolates = discriminator(interpolates)
    # 對所有輸出計算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # 計算梯度的範數
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    # 返回梯度懲罰
    return torch.mean((gradient_norm - 1) ** 2)


# 訓練WGAN-GP模型
def train_wgan_gp(
        generator: nn.Module,
        discriminator: nn.Module,
        dataloader: DataLoader,
        latent_dim: int,
        n_critic: int = N_CRITIC,
        lambda_gp: float = LAMBDA_GP,
        epochs: int = EPOCHS,
        device: torch.device = device
) -> Tuple[List[float], List[float]]:
    """訓練WGAN-GP模型"""
    # 優化器
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.9))

    # 記錄訓練過程的損失
    g_losses = []
    d_losses = []

    # 訓練迴圈
    for epoch in range(epochs):
        total_g_loss = 0.0
        total_d_loss = 0.0

        for i, real_samples in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            real_samples = real_samples.to(device)
            batch_size = real_samples.size(0)

            # 訓練判別器
            for _ in range(n_critic):
                optimizer_d.zero_grad()

                # 生成假樣本
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_samples = generator(z)

                # 計算判別器對真實樣本和生成樣本的評分
                real_validity = discriminator(real_samples)
                fake_validity = discriminator(fake_samples.detach())

                # 計算Wasserstein損失
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

                # 計算梯度懲罰
                gp = gradient_penalty(discriminator, real_samples, fake_samples, device)

                # 添加梯度懲罰
                d_loss = d_loss + lambda_gp * gp

                # 反向傳播和優化
                d_loss.backward()
                optimizer_d.step()

                total_d_loss += d_loss.item()

            # 訓練生成器
            optimizer_g.zero_grad()

            # 生成新的假樣本
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_samples = generator(z)

            # 計算判別器對生成樣本的評分
            fake_validity = discriminator(fake_samples)

            # 計算生成器損失
            g_loss = -torch.mean(fake_validity)

            # 反向傳播和優化
            g_loss.backward()
            optimizer_g.step()

            total_g_loss += g_loss.item()

        # 計算平均損失
        avg_g_loss = total_g_loss / len(dataloader)
        avg_d_loss = total_d_loss / (len(dataloader) * n_critic)

        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        # 輸出訓練進度
        print(f"Epoch {epoch + 1}/{epochs}, G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

    return g_losses, d_losses


# 生成序列
def generate_sequences(
        generator: nn.Module,
        latent_dim: int,
        num_sequences: int,
        device: torch.device
) -> List[str]:
    """使用生成器生成mRNA序列"""
    generator.eval()

    with torch.no_grad():
        # 生成潛在向量
        z = torch.randn(num_sequences, latent_dim, device=device)
        # 生成樣本
        fake_samples = generator(z)

        # 將生成的樣本轉換為numpy數組
        fake_samples_np = fake_samples.cpu().numpy()

        # 將樣本轉換為序列
        sequences = []
        for i in range(num_sequences):
            sequence = onehot_to_sequence(fake_samples_np[i])
            sequences.append(sequence)

    return sequences


# 繪製訓練損失曲線
def plot_losses(g_losses: List[float], d_losses: List[float]) -> None:
    """繪製訓練損失曲線"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig("wgan_gp_losses.png")
    plt.show()


# 分析核甘酸分佈
def analyze_nucleotide_distribution(real_sequences: List[str], generated_sequences: List[str]) -> None:
    """分析真實序列和生成序列的核甘酸分佈，並使用相同的色階範圍"""
    # 計算真實序列中每個位置的核甘酸分佈
    real_pos_counts = []
    for pos in range(TARGET_LENGTH):
        pos_count = Counter(seq[pos] for seq in real_sequences)
        real_pos_counts.append(pos_count)

    # 計算生成序列中每個位置的核甘酸分佈
    gen_pos_counts = []
    for pos in range(TARGET_LENGTH):
        pos_count = Counter(seq[pos] for seq in generated_sequences)
        gen_pos_counts.append(pos_count)

    # 真實序列核甘酸分佈
    real_data = np.zeros((len(NUCLEOTIDES), TARGET_LENGTH))
    for i, n in enumerate(NUCLEOTIDES):
        for pos in range(TARGET_LENGTH):
            real_data[i, pos] = real_pos_counts[pos][n] / len(real_sequences)

    # 生成序列核甘酸分佈
    gen_data = np.zeros((len(NUCLEOTIDES), TARGET_LENGTH))
    for i, n in enumerate(NUCLEOTIDES):
        for pos in range(TARGET_LENGTH):
            gen_data[i, pos] = gen_pos_counts[pos][n] / len(generated_sequences)

    # 找出最小與最大值，統一顏色範圍
    all_data = np.concatenate([real_data, gen_data])
    vmin = all_data.min()
    vmax = all_data.max()

    # 繪圖
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # 真實序列
    ax = axes[0]
    im = ax.imshow(real_data, cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title("Real Sequences Nucleotide Distribution")
    ax.set_yticks(range(len(NUCLEOTIDES)))
    ax.set_yticklabels(NUCLEOTIDES)
    ax.set_xlabel("Position")
    ax.set_ylabel("Nucleotide")
    plt.colorbar(im, ax=ax, label="Frequency")

    # 生成序列
    ax = axes[1]
    im = ax.imshow(gen_data, cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title("Generated Sequences Nucleotide Distribution")
    ax.set_yticks(range(len(NUCLEOTIDES)))
    ax.set_yticklabels(NUCLEOTIDES)
    ax.set_xlabel("Position")
    ax.set_ylabel("Nucleotide")
    plt.colorbar(im, ax=ax, label="Frequency")

    plt.tight_layout()
    plt.savefig("nucleotide_distribution.png")
    plt.show()


# 使用t-SNE或PCA進行序列可視化
def visualize_sequences(real_onehot: np.ndarray, generated_onehot: np.ndarray, method: str = 'pca') -> None:
    """使用降維方法可視化真實序列和生成序列"""
    # 將序列平坦化以進行降維
    real_flat = real_onehot.reshape(real_onehot.shape[0], -1)
    generated_flat = generated_onehot.reshape(generated_onehot.shape[0], -1)

    # 結合真實和生成的序列
    combined = np.vstack([real_flat, generated_flat])

    # 標籤: 0 for real, 1 for generated
    labels = np.array([0] * len(real_flat) + [1] * len(generated_flat))

    # 應用降維
    if method.lower() == 'tsne':
        reduced = TSNE(n_components=2, random_state=42).fit_transform(combined)
    else:  # PCA
        reduced = PCA(n_components=2, random_state=42).fit_transform(combined)

    # 繪製可視化結果
    plt.figure(figsize=(10, 8))

    plt.scatter(reduced[labels == 0, 0], reduced[labels == 0, 1], c='blue', alpha=0.5, label='Real')
    plt.scatter(reduced[labels == 1, 0], reduced[labels == 1, 1], c='red', alpha=0.5, label='Generated')

    plt.title(f"Sequence Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{method}_visualization.png")
    plt.show()


def compute_sequence_entropy(sequences: List[str]) -> np.ndarray:
    """計算每個位置的Shannon entropy作為序列多樣性指標"""
    entropies = np.zeros(TARGET_LENGTH)
    for pos in range(TARGET_LENGTH):
        counts = Counter(seq[pos] for seq in sequences)
        total = sum(counts.values())
        probs = [count / total for count in counts.values()]
        entropies[pos] = -sum(p * np.log2(p) for p in probs if p > 0)
    return entropies


def plot_entropy(real_sequences: List[str], generated_sequences: List[str]) -> None:
    real_entropy = compute_sequence_entropy(real_sequences)
    gen_entropy = compute_sequence_entropy(generated_sequences)

    plt.figure(figsize=(12, 6))
    plt.plot(real_entropy, label='Real', color='blue')
    plt.plot(gen_entropy, label='Generated', color='red')
    plt.xlabel("Position")
    plt.ylabel("Shannon Entropy")
    plt.title("Sequence Diversity (Entropy per Position)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sequence_entropy.png")
    plt.show()


def count_kmers(sequences: List[str], k: int) -> Counter:
    kmers = Counter()
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmers[kmer] += 1
    return kmers


def plot_top_kmers(real_sequences: List[str], generated_sequences: List[str], k: int = 3, top_n: int = 20) -> None:
    real_kmers = count_kmers(real_sequences, k)
    gen_kmers = count_kmers(generated_sequences, k)

    # 取出最常見的kmer
    all_kmers = list((real_kmers + gen_kmers).keys())
    top_kmers = sorted(all_kmers, key=lambda x: real_kmers[x] + gen_kmers[x], reverse=True)[:top_n]

    real_freqs = [real_kmers[kmer] for kmer in top_kmers]
    gen_freqs = [gen_kmers[kmer] for kmer in top_kmers]

    x = np.arange(top_n)
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, real_freqs, width, label='Real')
    plt.bar(x + width/2, gen_freqs, width, label='Generated')
    plt.xticks(x, top_kmers, rotation=45)
    plt.xlabel(f"{k}-mer")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} {k}-mer Frequencies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"top_{k}mer_frequencies.png")
    plt.show()


# 主函數
def main():
    # 處理FASTA文件
    file_path = "mature.fa"
    sequences, species = process_fasta_file(file_path)
    print(f"處理後的序列數量: {len(sequences)}")

    # 將序列轉換為one-hot編碼
    onehot_sequences = [sequence_to_onehot(seq) for seq in sequences]

    # 創建數據集和數據加載器
    dataset = MRNADataset(sequences)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 創建模型
    generator = Generator(LATENT_DIM, TARGET_LENGTH, len(NUCLEOTIDES)).to(device)
    discriminator = Discriminator(TARGET_LENGTH, len(NUCLEOTIDES)).to(device)

    print("生成器模型：")
    print(generator)
    print("\n判別器模型：")
    print(discriminator)

    # 訓練模型
    g_losses, d_losses = train_wgan_gp(generator, discriminator, dataloader, LATENT_DIM)

    # 繪製訓練損失
    plot_losses(g_losses, d_losses)

    # 生成序列
    num_gen_sequences = 1000
    generated_sequences = generate_sequences(generator, LATENT_DIM, num_gen_sequences, device)

    # 將生成的序列轉換為one-hot編碼
    generated_onehot = np.array([sequence_to_onehot(seq) for seq in generated_sequences])

    # 隨機抽樣一些真實序列進行比較
    sample_size = min(1000, len(sequences))
    sampled_indices = np.random.choice(len(sequences), sample_size, replace=False)
    sampled_sequences = [sequences[i] for i in sampled_indices]
    sampled_onehot = np.array([sequence_to_onehot(seq) for seq in sampled_sequences])

    # 分析核甘酸分佈
    analyze_nucleotide_distribution(sampled_sequences, generated_sequences)

    # 可視化序列
    visualize_sequences(sampled_onehot, generated_onehot, method='pca')
    visualize_sequences(sampled_onehot, generated_onehot, method='tsne')

    # 分析序列多樣性
    plot_entropy(sampled_sequences, generated_sequences)

    # 分析 k-mer 頻率（你可以改變 k 和 top_n）
    plot_top_kmers(sampled_sequences, generated_sequences, k=3, top_n=20)

    # 保存生成的序列
    with open("generated_sequences.fa", "w") as f:
        for i, seq in enumerate(generated_sequences):
            f.write(f">generated_sequence_{i + 1}\n{seq}\n")

    print("訓練和評估完成!")


if __name__ == "__main__":
    main()
