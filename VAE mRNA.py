import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from Bio import SeqIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from umap import UMAP
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


########################
# 1. DATA PREPROCESSING
########################

class MiRNADataPreprocessor:
    species_interested = {'hsa': 'Human', 'mmu': 'Mouse'}

    def __init__(self, fasta_file, target_length=22, min_length=15, max_length=30):
        """
        Initialize the miRNA data preprocessor

        Args:
            fasta_file (str): Path to the miRNA FASTA file
            target_length (int): Target sequence length for padding/truncation
            min_length (int): Minimum acceptable sequence length
            max_length (int): Maximum acceptable sequence length
        """
        self.fasta_file = fasta_file
        self.target_length = target_length
        self.min_length = min_length
        self.max_length = max_length
        self.nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}

    def load_and_filter_sequences(self):
        """
        Load sequences from FASTA file and filter based on length and content

        Returns:
            list: Filtered sequence records
        """
        valid_records = []
        invalid_records = []

        for record in SeqIO.parse(self.fasta_file, "fasta"):
            species_name = record.description.split()[0].split('-')[0] if len(
                record.description.split()) > 1 else "Unknown"
            if species_name not in self.species_interested:
                continue

            seq = str(record.seq).upper().replace('T', 'U')

            # Check sequence length
            if len(seq) < self.min_length or len(seq) > self.max_length:
                invalid_records.append((record.id, "Length out of range"))
                continue

            # Check for invalid characters
            if not all(nuc in self.nucleotide_map for nuc in seq):
                invalid_records.append((record.id, "Invalid nucleotides"))
                continue

            # Extract species and miRNA family information from the ID
            species = record.id.split('-')[0]

            # Try to extract miRNA family (usually after miR- prefix)
            family_match = re.search(r'miR-(\d+)', record.id)
            family = family_match.group(1) if family_match else "unknown"

            valid_records.append({
                'id': record.id,
                'sequence': seq,
                'length': len(seq),
                'species': species,
                'family': family
            })

        print(f"Total sequences: {len(valid_records) + len(invalid_records)}")
        print(f"Valid sequences: {len(valid_records)}")
        print(f"Invalid sequences: {len(invalid_records)}")

        # Save invalid records for inspection
        if invalid_records:
            pd.DataFrame(invalid_records, columns=['id', 'reason']).to_csv('invalid_mirna_records.csv', index=False)

        return valid_records

    def standardize_length(self, sequences):
        """
        Standardize sequences to target length by padding or truncation

        Args:
            sequences (list): List of sequence dictionaries

        Returns:
            list: List of sequence dictionaries with standardized sequences
        """
        standardized_sequences = []

        for record in sequences:
            seq = record['sequence']
            original_length = len(seq)

            # Truncate if longer than target length (from right/3' end)
            if len(seq) > self.target_length:
                seq = seq[:self.target_length]

            # Pad with 'N' if shorter than target length (at right/3' end)
            elif len(seq) < self.target_length:
                seq = seq + 'N' * (self.target_length - len(seq))

            record['standardized_sequence'] = seq
            record['original_length'] = original_length
            standardized_sequences.append(record)

        return standardized_sequences

    def one_hot_encode(self, sequences):
        """
        One-hot encode the standardized sequences

        Args:
            sequences (list): List of sequence dictionaries

        Returns:
            tuple: (encoded_sequences, sequence_metadata)
        """
        # Number of unique nucleotides (A, C, G, U, N)
        n_nucleotides = len(self.nucleotide_map)
        encoded_sequences = np.zeros((len(sequences), self.target_length, n_nucleotides))

        for i, record in enumerate(sequences):
            seq = record['standardized_sequence']
            for j, nuc in enumerate(seq):
                encoded_sequences[i, j, self.nucleotide_map[nuc]] = 1

        # Prepare metadata DataFrame
        metadata = pd.DataFrame(sequences)

        return encoded_sequences, metadata

    def preprocess(self, folder='result'):
        """
        Execute the full preprocessing pipeline
        Args:
            folder (str): Directory to save results

        Returns:
            tuple: (encoded_sequences, sequence_metadata)
        """
        raw_sequences = self.load_and_filter_sequences()
        standardized_sequences = self.standardize_length(raw_sequences)
        encoded_sequences, metadata = self.one_hot_encode(standardized_sequences)

        # Save metadata for later analysis
        if not os.path.exists(folder):
            os.mkdir(folder)
        metadata.to_csv(f'{folder}/mirna_metadata.csv', index=False)

        print("Sequence length distribution:")
        print(metadata['original_length'].describe())

        plt.figure(figsize=(10, 5))
        plt.hist(metadata['original_length'], bins=20)
        plt.xlabel('Sequence Length')
        plt.ylabel('Count')
        plt.title('Distribution of Original miRNA Sequence Lengths')
        plt.savefig(f'{folder}/sequence_length_distribution.png')

        return encoded_sequences, metadata


###########################
# 2. DATASET AND DATALOADER
###########################

class MiRNADataset(Dataset):
    def __init__(self, encoded_sequences):
        """
        Dataset for miRNA sequences

        Args:
            encoded_sequences (numpy.ndarray): One-hot encoded sequences
        """
        self.data = torch.FloatTensor(encoded_sequences)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


###################
# 3. VAE MODEL
###################

class CNNEncoder(nn.Module):
    def __init__(self, seq_length=22, n_nucleotides=5, latent_dim=32):
        """
        CNN encoder for the VAE

        Args:
            seq_length (int): Length of the input sequences
            n_nucleotides (int): Number of nucleotide types (A, C, G, U, N)
            latent_dim (int): Dimension of the latent space
        """
        super(CNNEncoder, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(n_nucleotides, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc_size = 128 * seq_length
        self.fc1 = nn.Linear(self.fc_size, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        # Input shape: batch_size x seq_length x n_nucleotides
        # Convert to: batch_size x n_nucleotides x seq_length for CNN
        x = x.permute(0, 2, 1)

        # Apply CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layer
        x = F.relu(self.fc1(x))

        # Get mu and logvar for VAE
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class BiLSTMEncoder(nn.Module):
    def __init__(self, seq_length=22, n_nucleotides=5, latent_dim=32):
        """
        BiLSTM encoder for the VAE

        Args:
            seq_length (int): Length of the input sequences
            n_nucleotides (int): Number of nucleotide types (A, C, G, U, N)
            latent_dim (int): Dimension of the latent space
        """
        super(BiLSTMEncoder, self).__init__()

        # BiLSTM layer
        self.lstm = nn.LSTM(input_size=n_nucleotides,
                            hidden_size=64,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)

        # Fully connected layers
        self.fc1 = nn.Linear(seq_length * 128, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        # Apply BiLSTM
        x, _ = self.lstm(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # FC layer
        x = F.relu(self.fc1(x))

        # Get mu and logvar for VAE
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=32, seq_length=22, n_nucleotides=5):
        """
        Decoder for the VAE

        Args:
            latent_dim (int): Dimension of the latent space
            seq_length (int): Length of the output sequences
            n_nucleotides (int): Number of nucleotide types (A, C, G, U, N)
        """
        super(Decoder, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, seq_length * 64)

        # Reshape parameters
        self.seq_length = seq_length
        self.hidden_dim = 64

        # Upsampling layers
        self.upsample1 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.upsample2 = nn.ConvTranspose1d(32, n_nucleotides, kernel_size=3, padding=1)

    def forward(self, z):
        # Fully connected layers
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))

        # Reshape for convolution
        x = x.view(-1, self.hidden_dim, self.seq_length)

        # Upsampling
        x = F.relu(self.upsample1(x))
        x = self.upsample2(x)

        # Convert back to batch_size x seq_length x n_nucleotides
        x = x.permute(0, 2, 1)

        # Apply softmax to get nucleotide probabilities
        x = F.softmax(x, dim=2)

        return x


class MiRNAVAE(nn.Module):
    def __init__(self, encoder_type='cnn', seq_length=22, n_nucleotides=5, latent_dim=32):
        """
        VAE for miRNA sequences

        Args:
            encoder_type (str): Type of encoder ('cnn' or 'bilstm')
            seq_length (int): Length of the sequences
            n_nucleotides (int): Number of nucleotide types (A, C, G, U, N)
            latent_dim (int): Dimension of the latent space
        """
        super(MiRNAVAE, self).__init__()

        # Choose encoder based on encoder_type
        if encoder_type == 'cnn':
            self.encoder = CNNEncoder(seq_length, n_nucleotides, latent_dim)
        elif encoder_type == 'bilstm':
            self.encoder = BiLSTMEncoder(seq_length, n_nucleotides, latent_dim)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        self.decoder = Decoder(latent_dim, seq_length, n_nucleotides)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick

        Args:
            mu (torch.Tensor): Mean of the latent distribution
            logvar (torch.Tensor): Log variance of the latent distribution

        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_x = self.decoder(z)

        return recon_x, mu, logvar

    def encode(self, x):
        """
        Encode input to latent representation

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Latent representation (mu)
        """
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu

    def decode(self, z):
        """
        Decode latent representation to sequence

        Args:
            z (torch.Tensor): Latent representation

        Returns:
            torch.Tensor: Reconstructed sequence probabilities
        """
        with torch.no_grad():
            recon_x = self.decoder(z)
        return recon_x


###################
# 4. TRAINING UTILS
###################

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function

    Args:
        recon_x (torch.Tensor): Reconstructed data
        x (torch.Tensor): Original data
        mu (torch.Tensor): Mean of the latent distribution
        logvar (torch.Tensor): Log variance of the latent distribution
        beta (float): Weight of the KL divergence term

    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence)
    """
    # Reconstruction loss (cross-entropy for one-hot encoded sequences)
    recon_loss = -torch.sum(x * torch.log(recon_x + 1e-10), dim=[1, 2]).mean()

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    # Total loss
    total_loss = recon_loss + beta * kl_div

    return total_loss, recon_loss, kl_div


def train_epoch(model, dataloader, optimizer, device, beta=1.0):
    """
    Train the model for one epoch

    Args:
        model (nn.Module): VAE model
        dataloader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device for computation
        beta (float): Weight of the KL divergence term

    Returns:
        tuple: (average_loss, average_recon_loss, average_kl_div)
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_div = 0

    for batch in dataloader:
        # Move data to device
        batch = batch.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar = model(batch)

        # Compute loss
        loss, recon_loss, kl_div = vae_loss(recon_batch, batch, mu, logvar, beta)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_div += kl_div.item()

    # Compute averages
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_div = total_kl_div / len(dataloader)

    return avg_loss, avg_recon_loss, avg_kl_div


def validate(model, dataloader, device, beta=1.0):
    """
    Validate the model

    Args:
        model (nn.Module): VAE model
        dataloader (DataLoader): Validation data loader
        device (torch.device): Device for computation
        beta (float): Weight of the KL divergence term

    Returns:
        tuple: (average_loss, average_recon_loss, average_kl_div)
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_div = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            batch = batch.to(device)

            # Forward pass
            recon_batch, mu, logvar = model(batch)

            # Compute loss
            loss, recon_loss, kl_div = vae_loss(recon_batch, batch, mu, logvar, beta)

            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_div += kl_div.item()

    # Compute averages
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_div = total_kl_div / len(dataloader)

    return avg_loss, avg_recon_loss, avg_kl_div


def train_vae(model, train_loader, val_loader, optimizer, device, epochs=100,
              beta_schedule=None, early_stopping_patience=10, save_folder='result'):
    """
    Train the VAE model

    Args:
        model (nn.Module): VAE model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device for computation
        epochs (int): Number of epochs
        beta_schedule (dict, optional): Schedule for beta annealing
        early_stopping_patience (int): Number of epochs to wait before early stopping
        save_folder (str): Directory to save the model

    Returns:
        dict: Training history
    """
    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_div': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_div': []
    }

    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Initialize beta
    beta = 0.0 if beta_schedule else 1.0

    for epoch in range(epochs):
        # Update beta according to schedule
        if beta_schedule:
            if epoch < beta_schedule['warmup_epochs']:
                beta = beta_schedule['start'] + (beta_schedule['end'] - beta_schedule['start']) * \
                       (epoch / beta_schedule['warmup_epochs'])
            else:
                beta = beta_schedule['end']

        # Train for one epoch
        train_loss, train_recon_loss, train_kl_div = train_epoch(model, train_loader, optimizer, device, beta)

        # Validate
        val_loss, val_recon_loss, val_kl_div = validate(model, val_loader, device, beta)

        # Update history
        history['train_loss'].append(train_loss)
        history['train_recon_loss'].append(train_recon_loss)
        history['train_kl_div'].append(train_kl_div)
        history['val_loss'].append(val_loss)
        history['val_recon_loss'].append(val_recon_loss)
        history['val_kl_div'].append(val_kl_div)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}, Beta: {beta:.3f}")
        print(f"Train Loss: {train_loss:.4f}, Recon Loss: {train_recon_loss:.4f}, KL Div: {train_kl_div:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Recon Loss: {val_recon_loss:.4f}, KL Div: {val_kl_div:.4f}")

        # Check for improvement
        if val_loss < best_val_loss and epoch > 10:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'beta': beta
            }, os.path.join(save_folder, 'mirna_vae_model.pt'))
            print(f"Model saved at epoch {epoch + 1}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history['train_recon_loss'], label='Train')
    plt.plot(history['val_recon_loss'], label='Validation')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history['train_kl_div'], label='Train')
    plt.plot(history['val_kl_div'], label='Validation')
    plt.title('KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')

    return history


###################
# 5. ANALYSIS UTILS
###################

def extract_latent_features(model, dataloader, device):
    """
    Extract latent features for all sequences

    Args:
        model (nn.Module): Trained VAE model
        dataloader (DataLoader): Data loader
        device (torch.device): Device for computation

    Returns:
        tuple: (latent_features, reconstruction_errors)
    """
    model.eval()
    latent_features = []
    reconstruction_errors = []

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            batch = batch.to(device)

            # Encode
            mu, logvar = model.encoder(batch)

            # Reparameterize
            z = model.reparameterize(mu, logvar)

            # Decode
            recon_batch = model.decoder(z)

            # Compute reconstruction error for each sample
            recon_error = -torch.sum(batch * torch.log(recon_batch + 1e-10), dim=[1, 2])

            # Store latent features and reconstruction errors
            latent_features.append(mu.cpu().numpy())
            reconstruction_errors.append(recon_error.cpu().numpy())

    # Concatenate batches
    latent_features = np.concatenate(latent_features, axis=0)
    reconstruction_errors = np.concatenate(reconstruction_errors, axis=0)

    return latent_features, reconstruction_errors


def dimensionality_reduction(latent_features, method='pca', n_components=2):
    """
    Perform dimensionality reduction on latent features

    Args:
        latent_features (numpy.ndarray): Latent features
        method (str): Method for dimensionality reduction ('pca' or 'umap')
        n_components (int): Number of components for dimensionality reduction

    Returns:
        numpy.ndarray: Reduced features
    """
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(latent_features)

    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'umap':
        reducer = UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")

    reduced_features = reducer.fit_transform(scaled_features)

    # If PCA, print explained variance
    if method == 'pca':
        explained_variance = reducer.explained_variance_ratio_
        print(f"Explained variance by {n_components} components: {sum(explained_variance):.4f}")
        for i, var in enumerate(explained_variance):
            print(f"Component {i + 1}: {var:.4f}")

    return reduced_features


def visualize_latent_space(reduced_features, metadata, color_by='species',
                           title='Latent Space Visualization', folder='result'):
    """
    Visualize the latent space

    Args:
        reduced_features (numpy.ndarray): Reduced features
        metadata (pandas.DataFrame): Sequence metadata
        color_by (str): Column name to color points by
        title (str): Title for the plot
    """
    plt.figure(figsize=(12, 10))

    # Get unique values for coloring
    unique_values = metadata[color_by].unique()

    # If too many unique values, limit to top N
    max_categories = 20
    if len(unique_values) > max_categories:
        value_counts = metadata[color_by].value_counts()
        top_values = value_counts.index[:max_categories]
        mask = metadata[color_by].isin(top_values)
        reduced_features_plot = reduced_features[mask]
        color_values = metadata.loc[mask, color_by]
        print(f"Limiting visualization to top {max_categories} {color_by} categories")
    else:
        reduced_features_plot = reduced_features
        color_values = metadata[color_by]

    # Plot
    scatter = plt.scatter(reduced_features_plot[:, 0], reduced_features_plot[:, 1],
                          c=pd.factorize(color_values)[0], cmap='tab20', alpha=0.7, s=30)

    # Add legend
    if len(unique_values) <= 20:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
                           for i in range(len(pd.factorize(color_values)[1]))]
        plt.legend(legend_elements, pd.factorize(color_values)[1], title=color_by,
                   loc='best', bbox_to_anchor=(1, 1))

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(f'{folder}/latent_space_{color_by}.png')


def analyze_reconstruction_errors(reconstruction_errors, metadata):
    """
    Analyze sequences with high reconstruction errors

    Args:
        reconstruction_errors (numpy.ndarray): Reconstruction errors
        metadata (pandas.DataFrame): Sequence metadata

    Returns:
        pandas.DataFrame: Analysis results
    """
    # Add reconstruction errors to metadata
    metadata_with_errors = metadata.copy()
    metadata_with_errors['reconstruction_error'] = reconstruction_errors

    # Sort by reconstruction error (descending)
    sorted_data = metadata_with_errors.sort_values(by='reconstruction_error', ascending=False)

    # Define a function to calculate GC content
    def calc_gc_content(sequence):
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) if len(sequence) > 0 else 0

    # Add GC content to the data
    sorted_data['gc_content'] = sorted_data['sequence'].apply(calc_gc_content)

    # Analyze GC content distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(sorted_data['gc_content'], kde=True)
    plt.title('GC Content Distribution')
    plt.xlabel('GC Content')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x='gc_content', y='reconstruction_error', data=sorted_data, alpha=0.6)
    plt.title('Reconstruction Error vs GC Content')
    plt.xlabel('GC Content')
    plt.ylabel('Reconstruction Error')

    plt.tight_layout()
    plt.savefig('gc_content_analysis.png')

    # Analyze species distribution
    plt.figure(figsize=(12, 5))
    species_error = sorted_data.groupby('species')['reconstruction_error'].mean().sort_values(ascending=False)
    top_species = species_error.head(15)

    sns.barplot(x=top_species.index, y=top_species.values)
    plt.title('Average Reconstruction Error by Species (Top 15)')
    plt.xlabel('Species')
    plt.ylabel('Average Reconstruction Error')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('species_error_analysis.png')

    # Analyze sequence length vs reconstruction error
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='original_length', y='reconstruction_error', data=sorted_data, alpha=0.6)
    plt.title('Reconstruction Error vs Original Sequence Length')
    plt.xlabel('Original Sequence Length')
    plt.ylabel('Reconstruction Error')
    plt.tight_layout()
    plt.savefig('length_error_analysis.png')

    # Analyze family distribution
    if 'family' in sorted_data.columns:
        plt.figure(figsize=(12, 5))
        family_error = sorted_data.groupby('family')['reconstruction_error'].mean().sort_values(ascending=False)
        top_families = family_error.head(15)

        sns.barplot(x=top_families.index, y=top_families.values)
        plt.title('Average Reconstruction Error by miRNA Family (Top 15)')
        plt.xlabel('Family')
        plt.ylabel('Average Reconstruction Error')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('family_error_analysis.png')

    return sorted_data


def analyze_sequence_characteristics(sequences):
    """
    Analyze sequence characteristics of miRNAs

    Args:
        sequences (pandas.DataFrame): Sequence data with metadata

    Returns:
        dict: Analysis results
    """
    # Calculate nucleotide frequencies
    nucleotide_counts = {'A': 0, 'C': 0, 'G': 0, 'U': 0, 'N': 0}
    position_counts = {pos: {'A': 0, 'C': 0, 'G': 0, 'U': 0, 'N': 0}
                       for pos in range(max(sequences['original_length']))}

    total_nucleotides = 0

    for _, row in sequences.iterrows():
        seq = row['sequence']
        for i, nuc in enumerate(seq):
            if i < len(position_counts):
                position_counts[i][nuc] += 1
            nucleotide_counts[nuc] += 1
            total_nucleotides += 1

    # Calculate nucleotide frequencies
    nucleotide_freq = {nuc: count / total_nucleotides for nuc, count in nucleotide_counts.items()
                       if nuc != 'N'}

    # Calculate position-specific nucleotide frequencies
    position_freq = {}
    for pos, counts in position_counts.items():
        total_pos = sum(counts.values())
        if total_pos > 0:
            position_freq[pos] = {nuc: count / total_pos for nuc, count in counts.items()}

    # Plot nucleotide frequency
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(nucleotide_freq.keys(), nucleotide_freq.values())
    plt.title('Overall Nucleotide Frequency')
    plt.xlabel('Nucleotide')
    plt.ylabel('Frequency')

    # Plot position-specific nucleotide frequency for first 22 positions
    plt.subplot(1, 2, 2)
    positions = list(range(min(22, len(position_freq))))
    width = 0.2

    for i, nuc in enumerate(['A', 'C', 'G', 'U']):
        freqs = [position_freq[pos][nuc] for pos in positions]
        plt.bar([p + i * width for p in positions], freqs, width=width, label=nuc)

    plt.title('Position-Specific Nucleotide Frequency')
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nucleotide_frequency_analysis.png')

    results = {
        'nucleotide_freq': nucleotide_freq,
        'position_freq': position_freq
    }

    return results


def analyze_latent_dimensions(latent_features, metadata):
    """
    Analyze individual latent dimensions

    Args:
        latent_features (numpy.ndarray): Latent features
        metadata (pandas.DataFrame): Sequence metadata
    """
    # Number of dimensions to analyze
    n_dims = min(8, latent_features.shape[1])

    plt.figure(figsize=(15, 10))

    for i in range(n_dims):
        plt.subplot(2, 4, i + 1)

        # Add latent dimension to metadata
        metadata_with_latent = metadata.copy()
        metadata_with_latent[f'latent_dim_{i}'] = latent_features[:, i]

        # Plot GC content vs latent dimension
        sns.scatterplot(x='gc_content', y=f'latent_dim_{i}', data=metadata_with_latent, alpha=0.5)
        plt.title(f'Latent Dimension {i + 1} vs GC Content')
        plt.xlabel('GC Content')
        plt.ylabel(f'Latent Dim {i + 1}')

    plt.tight_layout()
    plt.savefig('latent_dimensions_analysis.png')


###################
# 6. MAIN WORKFLOW
###################

if __name__ == '__main__':
    # Parameters
    fasta_file = 'mature.fa'  # Path to miRBase mature miRNA FASTA file
    target_length = 22  # Target sequence length
    latent_dim = 100  # Latent space dimension
    batch_size = 64  # Batch size for training
    encoder_type = 'bilstm'  # Encoder type ('cnn' or 'bilstm')
    epochs = 500  # Number of training epochs

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Preprocess data
    print("Preprocessing miRNA sequences...")
    preprocessor = MiRNADataPreprocessor(fasta_file, target_length=target_length)
    encoded_sequences, metadata = preprocessor.preprocess()

    # 2. Split data
    X_train, X_test, _, test_indices = train_test_split(
        encoded_sequences,
        np.arange(len(encoded_sequences)),
        test_size=0.2,
        random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = MiRNADataset(X_train)
    test_dataset = MiRNADataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    full_loader = DataLoader(MiRNADataset(encoded_sequences), batch_size=batch_size, shuffle=False)

    # 3. Create and train VAE model
    print(f"Creating {encoder_type.upper()} VAE model with latent dimension {latent_dim}...")
    n_nucleotides = encoded_sequences.shape[2]  # Number of nucleotide types
    model = MiRNAVAE(encoder_type=encoder_type,
                     seq_length=target_length,
                     n_nucleotides=n_nucleotides,
                     latent_dim=latent_dim).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Define beta annealing schedule
    beta_schedule = {
        'start': 0.0,
        'end': 0.7,
        'warmup_epochs': 70
    }

    # Train model
    print("Training VAE model...")
    train_vae(model, train_loader, test_loader, optimizer, device, epochs=epochs,
              beta_schedule=beta_schedule, early_stopping_patience=100)

    # 4. Load best model
    checkpoint = torch.load('result/mirna_vae_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch'] + 1}")

    # 5. Extract latent features
    print("Extracting latent features...")
    latent_features, reconstruction_errors = extract_latent_features(model, full_loader, device)

    # # 6. Dimensionality reduction
    # print("Performing dimensionality reduction...")
    # # PCA
    # pca_features = dimensionality_reduction(latent_features, method='pca', n_components=2)
    # visualize_latent_space(pca_features, metadata, color_by='species',
    #                        title='PCA of Latent Space (Species)')
    # visualize_latent_space(pca_features, metadata, color_by='family',
    #                        title='PCA of Latent Space (miRNA Family)')

    # UMAP
    umap_features = dimensionality_reduction(latent_features, method='umap', n_components=2)
    visualize_latent_space(umap_features, metadata, color_by='species',
                           title='UMAP of Latent Space (Species)')
    visualize_latent_space(umap_features, metadata, color_by='family',
                           title='UMAP of Latent Space (miRNA Family)')

    # 7. Analyze reconstruction errors
    print("Analyzing reconstruction errors...")
    metadata_with_errors = analyze_reconstruction_errors(reconstruction_errors, metadata)

    # 8. Analyze sequence characteristics
    print("Analyzing sequence characteristics...")
    sequence_analysis = analyze_sequence_characteristics(metadata_with_errors)

    # 9. Analyze latent dimensions
    print("Analyzing latent dimensions...")
    analyze_latent_dimensions(latent_features, metadata_with_errors)

    print("Analysis complete. Results saved to output files.")
