"""
Graph construction for spatial and expression similarity networks.

This module builds k-nearest neighbor graphs from spatial coordinates 
and gene expression profiles for graph-based velocity estimation.
"""

import warnings
from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import muon as mu
import anndata as ad

try:
    import torch_geometric
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    warnings.warn("torch_geometric not available. Graph functionality will be limited.")

from ..config import GraphConfig


class GraphBuilder:
    """
    Builder for spatial and expression similarity graphs.
    
    This class constructs k-nearest neighbor graphs from spatial coordinates
    and gene expression profiles, supporting various distance metrics and
    neighbor selection strategies.
    
    Parameters
    ----------
    config : GraphConfig
        Configuration object containing graph construction parameters.
    """
    
    def __init__(self, config: GraphConfig):
        self.config = config
        
        # Validate torch_geometric availability for graph operations
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for graph construction. "
                            "Install with: pip install torch_geometric")
    
    def build_graphs(self, adata: mu.MuData) -> Tuple[Data, Data]:
        """
        Build both spatial and expression graphs from MuData.
        
        Parameters
        ----------
        adata : mu.MuData
            Multi-modal data object containing spatial coordinates and expression data.
            
        Returns
        -------
        Tuple[Data, Data]
            Spatial graph and expression graph as PyTorch Geometric Data objects.
        """
        # Build spatial graph
        if self._has_spatial_coordinates(adata):
            spatial_coords = self._extract_spatial_coordinates(adata)
            spatial_graph = self.build_spatial_graph(spatial_coords)
        else:
            warnings.warn("No spatial coordinates found. Creating empty spatial graph.")
            spatial_graph = self._create_empty_graph(adata.n_obs)
            
        # Build expression graph
        expression_features = self._extract_expression_features(adata)
        expression_graph = self.build_expression_graph(expression_features)
        
        return spatial_graph, expression_graph
    
    def build_spatial_graph(self, coordinates: np.ndarray) -> Data:
        """
        Build k-NN graph from spatial coordinates.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Spatial coordinates array of shape (n_cells, 2) for (x, y) positions.
            
        Returns
        -------
        Data
            PyTorch Geometric Data object representing the spatial graph.
        """
        n_cells = coordinates.shape[0]
        
        if self.config.spatial_method == "knn":
            # k-NN graph construction
            adjacency_matrix = self._compute_knn_graph(
                coordinates,
                k=self.config.n_neighbors_spatial,
                metric='euclidean'  # Spatial coordinates use Euclidean distance
            )
        elif self.config.spatial_method == "radius":
            # Radius-based graph construction
            if self.config.spatial_radius is None:
                raise ValueError("spatial_radius must be specified for radius method")
            adjacency_matrix = self._compute_radius_graph(
                coordinates,
                radius=self.config.spatial_radius
            )
        else:
            raise ValueError(f"Unknown spatial method: {self.config.spatial_method}")
            
        # Convert to PyTorch Geometric format
        edge_index, edge_attr = self._sparse_to_edge_format(adjacency_matrix)
        
        # Create node features (spatial coordinates)
        node_features = torch.from_numpy(coordinates.astype(np.float32))
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n_cells
        )
    
    def build_expression_graph(self, features: np.ndarray) -> Data:
        """
        Build k-NN graph from gene expression profiles.
        
        Parameters
        ----------
        features : np.ndarray
            Expression feature matrix of shape (n_cells, n_features).
            
        Returns
        -------
        Data
            PyTorch Geometric Data object representing the expression graph.
        """
        n_cells = features.shape[0]
        
        # Build k-NN graph based on expression similarity
        adjacency_matrix = self._compute_knn_graph(
            features,
            k=self.config.n_neighbors_expression,
            metric=self.config.expression_metric
        )
        
        # Convert to PyTorch Geometric format
        edge_index, edge_attr = self._sparse_to_edge_format(adjacency_matrix)
        
        # Use expression features as node features
        node_features = torch.from_numpy(features.astype(np.float32))
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=n_cells
        )
    
    def _compute_knn_graph(self, features: np.ndarray, k: int, metric: str) -> sparse.csr_matrix:
        """
        Compute k-nearest neighbor graph with specified distance metric.
        
        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        k : int
            Number of nearest neighbors.
        metric : str
            Distance metric ('euclidean', 'cosine', 'manhattan', etc.).
            
        Returns
        -------
        sparse.csr_matrix
            Adjacency matrix of the k-NN graph.
        """
        n_samples = features.shape[0]
        
        # Ensure k is reasonable
        k = min(k, n_samples - 1)
        
        # Use scikit-learn's NearestNeighbors for efficient k-NN search
        if metric == 'cosine':
            # For cosine similarity, use cosine distance
            nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
        else:
            nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
            
        nn.fit(features)
        distances, indices = nn.kneighbors(features)
        
        # Remove self-connections (first neighbor is always the point itself)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Build sparse adjacency matrix
        row_indices = np.repeat(np.arange(n_samples), k)
        col_indices = indices.flatten()
        
        # Convert distances to similarities/weights
        if metric == 'cosine':
            # For cosine distance, convert to similarity: sim = 1 - dist
            edge_weights = 1.0 - distances.flatten()
        else:
            # For other metrics, use Gaussian kernel: exp(-dist^2 / (2*sigma^2))
            sigma = np.median(distances.flatten())
            edge_weights = np.exp(-distances.flatten() ** 2 / (2 * sigma ** 2))
            
        # Create symmetric adjacency matrix
        adjacency_matrix = sparse.csr_matrix(
            (edge_weights, (row_indices, col_indices)),
            shape=(n_samples, n_samples)
        )
        
        # Make the graph undirected by taking the maximum of (i,j) and (j,i)
        adjacency_matrix = sparse.maximum(adjacency_matrix, adjacency_matrix.T)
        
        return adjacency_matrix
    
    def _compute_radius_graph(self, coordinates: np.ndarray, radius: float) -> sparse.csr_matrix:
        """
        Compute radius-based graph where edges connect points within a radius.
        
        Parameters
        ----------
        coordinates : np.ndarray
            Coordinate matrix of shape (n_samples, n_dims).
        radius : float
            Connection radius.
            
        Returns
        -------
        sparse.csr_matrix
            Adjacency matrix of the radius graph.
        """
        n_samples = coordinates.shape[0]
        
        # Compute pairwise distances
        distances = squareform(pdist(coordinates, 'euclidean'))
        
        # Create adjacency matrix where edges exist if distance <= radius
        adjacency_matrix = (distances <= radius).astype(np.float32)
        
        # Remove self-connections
        np.fill_diagonal(adjacency_matrix, 0)
        
        # Convert distances to weights using Gaussian kernel
        sigma = radius / 3.0  # Set sigma relative to radius
        weights = np.exp(-distances ** 2 / (2 * sigma ** 2))
        adjacency_matrix = adjacency_matrix * weights
        
        return sparse.csr_matrix(adjacency_matrix)
    
    def _sparse_to_edge_format(self, adjacency_matrix: sparse.csr_matrix) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert sparse adjacency matrix to PyTorch Geometric edge format.
        
        Parameters
        ----------
        adjacency_matrix : sparse.csr_matrix
            Sparse adjacency matrix.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Edge indices and edge attributes tensors.
        """
        # Convert to COO format for easy extraction of indices
        coo_matrix = adjacency_matrix.tocoo()
        
        # Extract edge indices
        edge_index = torch.from_numpy(
            np.vstack([coo_matrix.row, coo_matrix.col])
        ).long()
        
        # Extract edge weights
        edge_attr = torch.from_numpy(coo_matrix.data).float()
        
        return edge_index, edge_attr
    
    def _has_spatial_coordinates(self, adata: mu.MuData) -> bool:
        """Check if spatial coordinates are available."""
        required_coords = ['x_pixel', 'y_pixel']
        return all(coord in adata.obs.columns for coord in required_coords)
    
    def _extract_spatial_coordinates(self, adata: mu.MuData) -> np.ndarray:
        """Extract spatial coordinates from MuData object."""
        coords = adata.obs[['x_pixel', 'y_pixel']].values
        
        # Validate coordinates
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            raise ValueError("Spatial coordinates contain NaN or infinite values")
            
        return coords.astype(np.float32)
    
    def _extract_expression_features(self, adata: mu.MuData) -> np.ndarray:
        """
        Extract expression features for graph construction.
        
        Uses PCA or highly variable genes if available, otherwise raw expression.
        """
        rna_data = adata['rna']
        
        # Try to use existing PCA representation first
        if 'X_pca' in rna_data.obsm:
            features = rna_data.obsm['X_pca']
            if features.shape[1] > 50:  # Limit to first 50 PCs for efficiency
                features = features[:, :50]
        else:
            # Use raw expression data
            if sparse.issparse(rna_data.X):
                features = rna_data.X.toarray()
            else:
                features = rna_data.X
                
            # If too many genes, use highly variable genes if available
            if features.shape[1] > 2000:
                if 'highly_variable' in rna_data.var.columns:
                    hvg_mask = rna_data.var['highly_variable'].values
                    features = features[:, hvg_mask]
                else:
                    # Use top 2000 most variable genes
                    gene_var = np.var(features, axis=0)
                    top_var_idx = np.argsort(gene_var)[-2000:]
                    features = features[:, top_var_idx]
                    
        # Normalize features
        features = features.astype(np.float32)
        
        # Log transform if needed (check if data is count-like)
        if np.all(features >= 0) and np.all(features == np.round(features)):
            features = np.log1p(features)
            
        # Standardize features
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        return features
    
    def _create_empty_graph(self, n_nodes: int) -> Data:
        """Create an empty graph with no edges."""
        return Data(
            x=torch.zeros((n_nodes, 1), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0,), dtype=torch.float32),
            num_nodes=n_nodes
        )