import sys
sys.path.append('/work/LAS/weile-lab/howlader/cancer-net')
import torch_geometric.transforms as T
from cancernet.arch import PNet
from cancernet.util import ProgressBar, InMemoryLogger, get_roc
from cancernet import PnetDataSet, ReactomeNetwork
from cancernet.dataset import get_layer_maps
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

## Graph Transformer DataSet
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from typing import Dict, List, Tuple, Optional
import os

class PNetGraphTransformerDataset(Dataset):
    """
    Convert P-NET dataset to Graph Transformer format.
    Each patient becomes a gene interaction graph.
    """
    
    def __init__(
        self,
        pnet_dataset,  # Your existing PnetDataSet
        reactome_network,
        edge_construction_method='pathway_cooccurrence',
        edge_threshold=0.1,
        add_self_loops=False,
        transform=None,
        pre_transform=None
    ):
        self.pnet_dataset = pnet_dataset
        self.reactome_network = reactome_network
        self.edge_construction_method = edge_construction_method
        self.edge_threshold = edge_threshold
        self.add_self_loops = add_self_loops
        
        # Extract basic info from P-NET dataset
        self.num_genes = len(pnet_dataset.genes)  # Number of genes
        self.num_samples = len(pnet_dataset.y)  # Number of patients
        self.gene_features = pnet_dataset.num_features  # mut, amp, del (3 features)
        
        # Get gene names and create mapping
        self.gene_names = pnet_dataset.genes
        self.gene_to_idx = pnet_dataset.node_index
        
        # Build the biological gene interaction graph
        self.edge_index, self.edge_attr = self._build_gene_graph()
        
        print(f"Graph Dataset Info:")
        print(f"  - Samples: {self.num_samples}")
        print(f"  - Genes: {self.num_genes}")
        print(f"  - Gene Features: {self.gene_features}")
        print(f"  - Edges: {self.edge_index.shape[1]}")
        
        super().__init__(transform=transform, pre_transform=pre_transform)
    
    def _build_gene_graph(self):
        """Build gene-gene interaction graph from biological knowledge."""
        
        if self.edge_construction_method == 'pathway_cooccurrence':
            return self._build_pathway_cooccurrence_graph()
        elif self.edge_construction_method == 'correlation':
            return self._build_correlation_graph()
        elif self.edge_construction_method == 'hybrid':
            return self._build_hybrid_graph()
        else:
            raise ValueError(f"Unknown edge construction method: {self.edge_construction_method}")
    
    def _build_pathway_cooccurrence_graph(self):
        """Create edges between genes that co-occur in pathways."""
        edge_list = []
        edge_weights = []
        
        # Get pathway-gene relationships from ReactomeNetwork
        pathway_gene_dict = self._extract_pathway_genes_from_reactome()
        
        if not pathway_gene_dict:
            print("No pathway data found, falling back to correlation method")
            return self._build_correlation_graph()
        
        # Iterate through Reactome pathways
        for pathway_id, pathway_genes in pathway_gene_dict.items():
            # Find genes in this pathway that are also in our dataset
            pathway_gene_indices = []
            for gene in pathway_genes:
                if gene in self.gene_to_idx:
                    pathway_gene_indices.append(self.gene_to_idx[gene])
            
            # Create edges between all pairs of genes in this pathway
            for i, gene_idx1 in enumerate(pathway_gene_indices):
                for gene_idx2 in pathway_gene_indices[i+1:]:
                    edge_list.append([gene_idx1, gene_idx2])
                    edge_list.append([gene_idx2, gene_idx1])  # Undirected
                    edge_weights.extend([1.0, 1.0])
        
        # Remove duplicate edges and aggregate weights
        edge_dict = {}
        for (i, j), weight in zip(edge_list, edge_weights):
            if (i, j) in edge_dict:
                edge_dict[(i, j)] += weight
            else:
                edge_dict[(i, j)] = weight
        
        # Convert to tensors
        edges = list(edge_dict.keys())
        weights = list(edge_dict.values())
        
        if len(edges) == 0:
            # Fallback: create empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0,), dtype=torch.float)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).T
            edge_attr = torch.tensor(weights, dtype=torch.float)
        
        # Add self-loops if requested
        if self.add_self_loops:
            self_loop_edges = torch.arange(self.num_genes).repeat(2, 1)
            self_loop_weights = torch.ones(self.num_genes)
            edge_index = torch.cat([edge_index, self_loop_edges], dim=1)
            edge_attr = torch.cat([edge_attr, self_loop_weights])
        
        return edge_index, edge_attr
    
    def _extract_pathway_genes_from_reactome(self):
        """
        Extract pathway-gene relationships from ReactomeNetwork object.
        ReactomeNetwork.reactome.pathway_genes is a DataFrame with 'group' and 'gene' columns.
        """
        pathway_gene_dict = {}
        
        try:
            # Access the pathway_genes DataFrame from ReactomeNetwork
            if hasattr(self.reactome_network, 'reactome') and hasattr(self.reactome_network.reactome, 'pathway_genes'):
                pathway_genes_df = self.reactome_network.reactome.pathway_genes
                
                # Convert DataFrame to dictionary: {pathway_name: [list_of_genes]}
                for pathway_name, group_df in pathway_genes_df.groupby('group'):
                    genes_list = group_df['gene'].tolist()
                    pathway_gene_dict[pathway_name] = genes_list
                
                print(f"Successfully extracted {len(pathway_gene_dict)} pathways from ReactomeNetwork")
                return pathway_gene_dict
            
            # Fallback: direct access to pathway_genes if it exists
            elif hasattr(self.reactome_network, 'pathway_genes'):
                pathway_genes_df = self.reactome_network.pathway_genes
                if isinstance(pathway_genes_df, pd.DataFrame):
                    for pathway_name, group_df in pathway_genes_df.groupby('group'):
                        genes_list = group_df['gene'].tolist()
                        pathway_gene_dict[pathway_name] = genes_list
                    return pathway_gene_dict
            
        except Exception as e:
            print(f"Error extracting pathway-gene data: {e}")
        
        # Final fallback
        if not pathway_gene_dict:
            print("Warning: Could not extract pathway-gene relationships from ReactomeNetwork.")
            print("Will use correlation-based graph construction instead.")
            return {}
        
        return pathway_gene_dict
    
    def _build_correlation_graph(self):
        """Create edges based on gene expression correlation."""
        # Get gene data: shape [num_samples, num_genes, num_features]
        gene_data = self.pnet_dataset.x
        
        # Average across features to get [num_samples, num_genes]
        if len(gene_data.shape) == 3:
            gene_data = gene_data.mean(dim=2)  # Average mut, amp, del
        
        # Compute correlation matrix
        correlation_matrix = torch.corrcoef(gene_data.T).numpy()
        
        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        # Threshold correlation matrix
        correlation_matrix = np.abs(correlation_matrix)
        adjacency = (correlation_matrix > self.edge_threshold).astype(float)
        np.fill_diagonal(adjacency, 0)  # Remove self-loops initially
        
        # Convert to edge list
        edge_indices = np.where(adjacency > 0)
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        edge_attr = torch.tensor(correlation_matrix[edge_indices], dtype=torch.float)
        
        return edge_index, edge_attr
    
    def _build_hybrid_graph(self):
        """Combine pathway and correlation information."""
        # Get pathway edges
        pathway_edge_index, pathway_edge_attr = self._build_pathway_cooccurrence_graph()
        
        # Get correlation edges  
        corr_edge_index, corr_edge_attr = self._build_correlation_graph()
        
        # Combine edges (pathway edges get higher weight)
        combined_edge_index = torch.cat([pathway_edge_index, corr_edge_index], dim=1)
        
        # Weight pathway edges higher
        pathway_weights = pathway_edge_attr * 2.0
        combined_edge_attr = torch.cat([pathway_weights, corr_edge_attr])
        
        return combined_edge_index, combined_edge_attr
    
    def len(self):
        return self.num_samples
    
    def get(self, idx):
        """Get a single patient as a graph."""
        # Extract patient features: [num_genes, gene_features]
        x = self.pnet_dataset.x[idx]  # Shape: [num_genes, 3] (mut, amp, del)
        
        # Get label
        y = self.pnet_dataset.y[idx]
        
        # Create graph data object
        data = Data(
            x=x.float(),  # [num_genes, gene_features]
            edge_index=self.edge_index,  # [2, num_edges]
            edge_attr=self.edge_attr.unsqueeze(-1) if self.edge_attr.dim() == 1 else self.edge_attr,  # [num_edges, 1]
            y=y.long(),  # [1]
            sample_idx=torch.tensor(idx, dtype=torch.long)
        )
        
        return data
    
    def get_splits(self):
        """Get train/val/test splits from P-NET dataset."""
        train_indices = self.pnet_dataset.train_idx
        val_indices = self.pnet_dataset.valid_idx
        test_indices = self.pnet_dataset.test_idx
        
        return train_indices, val_indices, test_indices


# Utility function to create data loaders
def create_graph_data_loaders(graph_dataset, batch_size=2, num_workers=8):
    """Create data loaders for train/val/test splits."""
    from torch_geometric.data import DataLoader
    from torch.utils.data import Subset
    
    # Get split indices
    train_indices, val_indices, test_indices = graph_dataset.get_splits()
    
    # Create subset datasets
    train_dataset = Subset(graph_dataset, train_indices)
    val_dataset = Subset(graph_dataset, val_indices)
    test_dataset = Subset(graph_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Data Loaders Created:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    print(f"  Test: {len(test_indices)} samples")
    
    return train_loader, val_loader, test_loader

## Graph Transformer Architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool
import math

class BiologicalGraphTransformer(nn.Module):
    """
    Graph Transformer for biological pathway data.
    Uses attention mechanisms to model gene-gene interactions.
    """
    
    def __init__(
        self,
        num_genes,
        gene_features=3,  # mut, amp, del features from P-NET
        hidden_dim=256,
        num_heads=8,
        num_layers=6,
        num_classes=2,
        dropout=0.1,
        pooling='hierarchical',  # 'mean', 'max', 'attention', 'hierarchical'
        use_edge_attr=True
    ):
        super().__init__()
        
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pooling = pooling
        self.use_edge_attr = use_edge_attr
        
        print(f"Building Graph Transformer:")
        print(f"  - Genes: {num_genes}")
        print(f"  - Gene features: {gene_features}")
        print(f"  - Hidden dim: {hidden_dim}")
        print(f"  - Attention heads: {num_heads}")
        print(f"  - Layers: {num_layers}")
        print(f"  - Pooling: {pooling}")
        
        # Gene feature embedding
        self.gene_embedding = nn.Linear(gene_features, hidden_dim)
        
        # Positional encoding for genes (optional biological prior)
        self.use_pos_encoding = True
        if self.use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(num_genes, hidden_dim) * 0.1)
        
        # Graph Transformer layers
        self.transformer_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            self.transformer_layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=False,  # Average multi-head outputs
                    dropout=dropout,
                    edge_dim=1 if use_edge_attr else None,  # Edge attributes
                    beta=True  # Use gated attention
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Pooling mechanism
        if pooling == 'attention':
            self.attention_pool = AttentionPooling(hidden_dim)
            pooled_dim = hidden_dim
        elif pooling == 'hierarchical':
            self.hierarchical_pool = HierarchicalPooling(hidden_dim)
            pooled_dim = hidden_dim * 3  # mean + max + attention
        else:
            pooled_dim = hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Calculate model parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  - Total parameters: {total_params:,}")
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, batch):
        """
        Forward pass through the graph transformer.
        
        Args:
            batch: PyTorch Geometric batch object
                - x: [total_nodes, gene_features] 
                - edge_index: [2, total_edges]
                - edge_attr: [total_edges, 1]
                - batch: [total_nodes] - batch assignment
        """
        x, edge_index, edge_attr, batch_vec = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        # Initial gene embedding
        h = self.gene_embedding(x)  # [total_nodes, hidden_dim]
        
        # Add positional encoding if used
        if self.use_pos_encoding:
            # Get gene positions for each sample in batch
            batch_size = batch_vec.max().item() + 1
            pos_encodings = []
            
            for i in range(batch_size):
                mask = (batch_vec == i)
                num_genes_in_sample = mask.sum().item()
                pos_encodings.append(self.pos_encoding[:num_genes_in_sample])
            
            pos_enc = torch.cat(pos_encodings, dim=0)
            h = h + pos_enc
        
        # Apply Graph Transformer layers
        for i, (transformer, norm, dropout) in enumerate(zip(
            self.transformer_layers, self.layer_norms, self.dropouts
        )):
            # Graph attention
            if self.use_edge_attr and edge_attr is not None:
                # edge_weights = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr
                edge_weights = edge_attr  # preserve shape [num_edges, 1]
                h_new = transformer(h, edge_index, edge_weights)
            else:
                h_new = transformer(h, edge_index)
            
            # Residual connection + layer norm + dropout
            h = norm(h + h_new)
            h = dropout(h)
            h = F.relu(h)
        
        # Graph-level pooling
        if self.pooling == 'mean':
            graph_repr = global_mean_pool(h, batch_vec)
        elif self.pooling == 'max':
            graph_repr = global_max_pool(h, batch_vec)
        elif self.pooling == 'attention':
            graph_repr = self.attention_pool(h, batch_vec)
        elif self.pooling == 'hierarchical':
            graph_repr = self.hierarchical_pool(h, batch_vec)
        else:
            graph_repr = global_mean_pool(h, batch_vec)
        
        # Classification
        logits = self.classifier(graph_repr)
        
        return logits, h  # Return both predictions and node embeddings


class AttentionPooling(nn.Module):
    """Learnable attention-based pooling for graph-level representation."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, batch):
        # Compute attention weights
        attn_weights = self.attention(x)  # [total_nodes, 1]
        
        # Apply softmax per graph
        batch_size = batch.max().item() + 1
        pooled = []
        
        for i in range(batch_size):
            mask = (batch == i)
            graph_nodes = x[mask]  # [num_nodes_in_graph, hidden_dim]
            graph_attn = attn_weights[mask]  # [num_nodes_in_graph, 1]
            
            # Softmax attention weights for this graph
            graph_attn = F.softmax(graph_attn, dim=0)
            
            # Weighted sum
            graph_repr = torch.sum(graph_nodes * graph_attn, dim=0)  # [hidden_dim]
            pooled.append(graph_repr)
        
        return torch.stack(pooled)  # [batch_size, hidden_dim]


class HierarchicalPooling(nn.Module):
    """Combine multiple pooling strategies for richer graph representation."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_pool = AttentionPooling(hidden_dim)
    
    def forward(self, x, batch):
        # Different pooling methods
        mean_pool = global_mean_pool(x, batch)  # Average gene activity
        max_pool = global_max_pool(x, batch)    # Maximum gene activity
        attn_pool = self.attention_pool(x, batch)  # Attention-weighted genes
        
        # Concatenate all representations
        return torch.cat([mean_pool, max_pool, attn_pool], dim=1)


# Model factory function
def create_biological_graph_transformer(
    num_genes,
    gene_features=3,
    model_size='medium',
    num_classes=2,
    pooling='hierarchical'
):
    """
    Factory function to create Graph Transformer with predefined configurations.
    
    Args:
        num_genes: Number of genes in the dataset
        gene_features: Number of features per gene (3 for P-NET: mut, amp, del)
        model_size: 'small', 'medium', 'large'
        num_classes: Number of output classes
        pooling: Pooling strategy
    
    Returns:
        BiologicalGraphTransformer model
    """
    
    configs = {
        'small': {
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 3,
            'dropout': 0.1
        },
        'medium': {
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1
        },
        'large': {
            'hidden_dim': 512,
            'num_heads': 12,
            'num_layers': 8,
            'dropout': 0.15
        }
    }
    
    config = configs.get(model_size, configs['medium'])
    
    model = BiologicalGraphTransformer(
        num_genes=num_genes,
        gene_features=gene_features,
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        num_classes=num_classes,
        dropout=config['dropout'],
        pooling=pooling
    )
    
    return model


# ===================================================================
# TRAINER CLASS (Complete Implementation)
# ===================================================================

class GraphTransformerTrainer:
    """Complete training and evaluation pipeline for Graph Transformer."""
    
    def __init__(
        self,
        model,
        device=None,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=15,
        min_delta=1e-4,
        save_dir='./checkpoints'
    ):
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Move to GPU first
        model = model.to(self.device)

        # If multiple GPUs are available, wrap in DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(model, device_ids=[0, 1])
        else:
            self.model = model

        
        # Model setup
        # self.model = model.to(self.device)
        self.patience = patience
        self.min_delta = min_delta
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        print(f"Trainer initialized:")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Weight decay: {weight_decay}")
        print(f"  - Patience: {patience}")
        print(f"  - Save directory: {save_dir}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(batch)
            loss = self.criterion(logits, batch.y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Collect metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader):
        """Evaluate on validation/test set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                logits, _ = self.model(batch)
                loss = self.criterion(logits, batch.y)
                
                total_loss += loss.item()
                
                # Predictions and probabilities
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, all_preds, all_labels, all_probs
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader, num_epochs=100, verbose=True):
        """Complete training loop with early stopping."""
        print(f"\n=== Starting Training ===")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        patience_counter = 0
        # start_time = time.time()
        
        for epoch in range(num_epochs):
            # epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, _, _, _ = self.evaluate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Check for improvement
            is_best = False
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                patience_counter = 0
                is_best = True
            else:
                patience_counter += 1
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Progress reporting
            # epoch_time = time.time() - epoch_start
            if verbose and (epoch % 5 == 0 or is_best):
                print(f"Epoch {epoch:3d}/{num_epochs} | "
                      f"Train(loss/acc): {train_loss:.4f}/{train_acc:.4f} | "
                      f"Val(loss/acc): {val_loss:.4f}/{val_acc:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"{'🏆' if is_best else '📈'}")
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                print(f"Best: Epoch {self.best_epoch} - Val Loss: {self.best_val_loss:.4f}, Val Acc: {self.best_val_acc:.4f}")
                break
        
        # Load best model
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            self.load_checkpoint(best_model_path)
        
        # total_time = time.time() - start_time
        # print(f"\n✅ Training completed in {total_time:.1f}s")
        print(f"Best model: Epoch {self.best_epoch} - Val Loss: {self.best_val_loss:.4f}, Val Acc: {self.best_val_acc:.4f}")
    

    def test(self, test_loader, verbose=True):
        """Final evaluation on test set."""
        print(f"\n=== Testing Best Model ===")

        test_loss, test_acc, test_preds, test_labels, test_probs = self.evaluate(test_loader)

        # Convert to numpy arrays for easier handling
        import numpy as np
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, f1_score, 
            precision_score, recall_score, classification_report, confusion_matrix
        )

        test_labels = np.array(test_labels)
        test_preds = np.array(test_preds)
        test_probs = np.array(test_probs)

        # Check if we have both classes
        unique_labels = np.unique(test_labels)

        # Initialize metrics
        auc_score = 0.0
        aupr_score = 0.0
        f1 = 0.0
        precision = 0.0
        recall = 0.0

        if len(unique_labels) >= 2:
            # We have both classes - can calculate all metrics
            try:
                # Get probabilities for class 1
                if test_probs.shape[1] >= 2:
                    test_probs_class1 = test_probs[:, 1]  # Probability of class 1
                else:
                    test_probs_class1 = test_probs[:, 0]  # Fallback

                # AUC-ROC
                auc_score = roc_auc_score(test_labels, test_probs_class1)

                # AUPR (Area Under Precision-Recall Curve)
                aupr_score = average_precision_score(test_labels, test_probs_class1)

            except Exception as e:
                print(f"Warning: Could not calculate AUC/AUPR: {e}")
                auc_score = 0.0
                aupr_score = 0.0
        else:
            print("Warning: Only one class present in test set - cannot calculate AUC/AUPR")

        # Calculate classification metrics (these work even with one class)
        try:
            # For binary classification, use 'binary' average
            if len(unique_labels) == 2:
                f1 = f1_score(test_labels, test_preds, average='binary')
                precision = precision_score(test_labels, test_preds, average='binary', zero_division=0)
                recall = recall_score(test_labels, test_preds, average='binary', zero_division=0)
            else:
                # For single class or multi-class, use macro average
                f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
                precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
                recall = recall_score(test_labels, test_preds, average='macro', zero_division=0)
        except Exception as e:
            print(f"Warning: Could not calculate F1/Precision/Recall: {e}")
            f1 = 0.0
            precision = 0.0
            recall = 0.0

        if verbose:
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")

            # Print metrics with appropriate handling for missing values
            if len(unique_labels) >= 2:
                print(f"Test AUC-ROC: {auc_score:.4f}")
                print(f"Test AUPR: {aupr_score:.4f}")
            else:
                print(f"Test AUC-ROC: N/A (only one class)")
                print(f"Test AUPR: N/A (only one class)")

            print(f"Test F1-Score: {f1:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")

            print(f"\nClassification Report:")
            print(classification_report(test_labels, test_preds, labels=unique_labels, zero_division=0))

            print(f"\nConfusion Matrix:")
            print(confusion_matrix(test_labels, test_preds))

            # Additional info
            print(f"\nDataset Info:")
            print(f"  Total samples: {len(test_labels)}")
            print(f"  Unique classes: {unique_labels}")
            if len(unique_labels) > 1:
                print(f"  Class distribution: {dict(zip(unique_labels, np.bincount(test_labels)))}")
            else:
                print(f"  Class distribution: {dict(zip(unique_labels, [len(test_labels)]))}")

        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_auc': auc_score,
            'test_aupr': aupr_score,
            'test_f1': f1,
            'test_precision': precision,
            'test_recall': recall,
            'predictions': test_preds,
            'labels': test_labels,
            'probabilities': test_probs
        }
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(len(self.history['train_loss']))
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        ax1.axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7, label='Best Model')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        ax2.axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7, label='Best Model')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        ax3.plot(epochs, self.history['learning_rates'], 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Best metrics summary
        ax4.text(0.1, 0.8, f"Best Epoch: {self.best_epoch}", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f"Best Val Loss: {self.best_val_loss:.4f}", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f"Best Val Acc: {self.best_val_acc:.4f}", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.5, f"Total Epochs: {len(epochs)}", fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Training Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plot saved to {save_path}")
        
        plt.show()
        

        
        
# ===================================================================
# COMPLETE PIPELINE FUNCTION
# ===================================================================

def train_graph_transformer_pipeline(
    graph_dataset,
    model_config=None,
    training_config=None,
    device=None
):
    """
    Complete pipeline to train Graph Transformer on your data.
    
    Args:
        graph_dataset: PNetGraphTransformerDataset object
        model_config: Dict with model hyperparameters
        training_config: Dict with training hyperparameters
        device: Training device
    
    Returns:
        Dict with trained model, trainer, and results
    """
    
    # Default configurations
    if model_config is None:
        model_config = {
            'hidden_dim': 64,
            'num_heads': 2,
            'num_layers': 2,
            'dropout': 0.1,
            'pooling': 'hierarchical'
        }
    
    if training_config is None:
        training_config = {
            'batch_size': 2,
            'learning_rate': 5e-4,
            'weight_decay': 1e-4,
            'num_epochs': 5,
            'patience': 15
        }
    
    print("=== COMPLETE GRAPH TRANSFORMER PIPELINE ===\n")
    
    # Step 1: Create data loaders
    print("Step 1: Creating data loaders...")
    from torch_geometric.data import DataLoader
    from torch.utils.data import Subset
    
    train_indices, val_indices, test_indices = graph_dataset.get_splits()
    
    train_dataset = Subset(graph_dataset, train_indices)
    val_dataset = Subset(graph_dataset, val_indices)
    test_dataset = Subset(graph_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=training_config['batch_size'], shuffle=False)
    
    print(f"  ✓ Train: {len(train_indices)} samples")
    print(f"  ✓ Val: {len(val_indices)} samples")
    print(f"  ✓ Test: {len(test_indices)} samples")
    
    # Step 2: Create model
    print(f"\nStep 2: Creating model...")
    model = BiologicalGraphTransformer(
        num_genes=graph_dataset.num_genes,
        gene_features=graph_dataset.gene_features,
        num_classes=2,
        **model_config
    )
    
    # Step 3: Create trainer
    print(f"\nStep 3: Creating trainer...")
    trainer = GraphTransformerTrainer(
        model=model,
        device=device,
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        patience=training_config['patience']
    )
    
    # Step 4: Train
    print(f"\nStep 4: Training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_config['num_epochs'],
        verbose=True
    )
    
    # Step 5: Test
    print(f"\nStep 5: Testing...")
    test_results = trainer.test(test_loader, verbose=True)
    
    # Step 6: Plot results
    print(f"\nStep 6: Plotting results...")
    trainer.plot_training_history()
    
    return {
        'model': model,
        'trainer': trainer,
        'graph_dataset': graph_dataset,
        'loaders': (train_loader, val_loader, test_loader),
        'test_results': test_results,
        'config': {
            'model': model_config,
            'training': training_config
        }
    }

#Working with dataset
import os
from cancernet.dataset import ReactomeNetwork, PnetDataSet
import torch_geometric.transforms as T

# ReactomeNetwork setup
reactome_kws = dict(
    reactome_base_dir=os.path.join("../data", "reactome"),
    relations_file_name="ReactomePathwaysRelation.txt",
    pathway_names_file_name="ReactomePathways.txt",
    pathway_genes_file_name="ReactomePathways.gmt",
)
reactome = ReactomeNetwork(reactome_kws)

# P-NET Dataset setup
prostate_root = os.path.join("../data", "prostate")
dataset = PnetDataSet(
    root=prostate_root,
    name="prostate_graph_humanbase",
    edge_tol=0.5,
    pre_transform=T.Compose([
        T.GCNNorm(add_self_loops=False), 
        T.ToSparseTensor(remove_edge_index=False)
    ])
)

# Load splits
splits_root = os.path.join(prostate_root, "splits")
dataset.split_index_by_file(
    train_fp=os.path.join(splits_root, "training_set_0.csv"),
    valid_fp=os.path.join(splits_root, "validation_set.csv"),
    test_fp=os.path.join(splits_root, "test_set.csv"),
)


# Create graph dataset from your P-NET data
graph_dataset = PNetGraphTransformerDataset(
    pnet_dataset=dataset,        # Your existing PnetDataSet
    reactome_network=reactome,   # Your existing ReactomeNetwork
    edge_construction_method='pathway_cooccurrence'
)

# Run complete training pipeline
results = train_graph_transformer_pipeline(
    graph_dataset=graph_dataset,
    model_config={
        'hidden_dim': 64,
        'num_heads': 2, 
        'num_layers': 2,
        'dropout': 0.1,
        'pooling': 'hierarchical'
    },
    training_config={
        'batch_size': 2,           # Start small for graph data
        'learning_rate': 5e-4,
        'weight_decay': 1e-4,
        'num_epochs': 5,
        'patience': 15
    }
)


print(f"\n=== Graph Transformer Test Results ===")
print(f"Test Accuracy: {results['test_results']['test_accuracy']:.4f}")
print(f"Test AUC-ROC: {results['test_results']['test_auc']:.4f}")
print(f"Test AUPR: {results['test_results']['test_aupr']:.4f}")
print(f"Test F1-Score: {results['test_results']['test_f1']:.4f}")
print(f"Test Precision: {results['test_results']['test_precision']:.4f}")
print(f"Test Recall: {results['test_results']['test_recall']:.4f}")
print(f"Test Loss: {results['test_results']['test_loss']:.4f}")