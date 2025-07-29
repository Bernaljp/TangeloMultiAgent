"""
Multi-modal data processing for Tangelo Velocity.

This module handles MuData object validation, preprocessing, and preparation
for graph-based velocity estimation. It includes ATAC-seq preprocessing
and integration with spatial transcriptomics data.
"""

import warnings
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import muon as mu
import scanpy as sc
import anndata as ad

from ..config import TangeloConfig
from .graph_builder import GraphBuilder


class MuDataProcessor:
    """
    Multi-modal data processor for Tangelo Velocity.
    
    This class handles validation, preprocessing, and graph construction
    for multi-modal single-cell data including RNA-seq and ATAC-seq.
    
    Parameters
    ----------
    config : TangeloConfig
        Configuration object containing preprocessing parameters.
    """
    
    def __init__(self, config: TangeloConfig):
        self.config = config
        self.graph_builder = GraphBuilder(config.graph)
        
        # Validation flags
        self._is_validated = False
        self._has_atac = False
        self._has_spatial = False
        
    def process_mudata(self, adata: mu.MuData) -> Dict[str, Any]:
        """
        Main processing pipeline for multi-modal data.
        
        Parameters
        ----------
        adata : mu.MuData
            Multi-modal annotated data object.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing processed components:
            - 'spliced': Spliced count matrix
            - 'unspliced': Unspliced count matrix
            - 'spatial_graph': Spatial proximity graph
            - 'expression_graph': Expression similarity graph
            - 'atac_mask': ATAC accessibility mask (if available)
            - 'node2vec_embeddings': Node embeddings (if enabled)
        """
        # Validate input data
        self.validate_mudata(adata)
        
        # Extract RNA data
        rna_data = adata['rna']
        spliced = torch.from_numpy(rna_data.layers['spliced'].toarray().astype(np.float32))
        unspliced = torch.from_numpy(rna_data.layers['unspliced'].toarray().astype(np.float32))
        
        # Build graphs
        spatial_graph, expression_graph = self.graph_builder.build_graphs(adata)
        
        # Prepare result dictionary
        result = {
            'spliced': spliced,
            'unspliced': unspliced,
            'spatial_graph': spatial_graph,
            'expression_graph': expression_graph,
        }
        
        # Process ATAC data if available and required
        if self._has_atac and self.config.regulatory.use_atac_masking:
            atac_mask = self.create_open_chromatin_mask(adata)
            result['atac_mask'] = atac_mask
            
        # Add node2vec embeddings if requested
        if self.config.graph.use_node2vec:
            from .node2vec_embedding import Node2VecEmbedding
            node2vec = Node2VecEmbedding(self.config.graph)
            
            # Combine spatial and expression graphs for embedding
            combined_graph = self._combine_graphs(spatial_graph, expression_graph)
            embeddings = node2vec.fit_transform(combined_graph)
            result['node2vec_embeddings'] = embeddings
            
        return result
    
    def validate_mudata(self, adata: mu.MuData) -> None:
        """
        Validate MuData object for required modalities and data quality.
        
        Parameters
        ----------
        adata : mu.MuData
            Multi-modal data object to validate.
            
        Raises
        ------
        ValueError
            If required modalities, layers, or data quality checks fail.
        """
        # Check basic structure
        if not isinstance(adata, mu.MuData):
            raise ValueError("Input must be a MuData object")
            
        if adata.n_obs == 0:
            raise ValueError("Data object is empty (0 observations)")
            
        # Check required RNA modality
        if 'rna' not in adata.mod:
            raise ValueError("MuData object must contain 'rna' modality")
            
        rna_data = adata['rna']
        
        # Check required RNA layers
        required_rna_layers = ['spliced', 'unspliced']
        missing_layers = [layer for layer in required_rna_layers 
                         if layer not in rna_data.layers]
        if missing_layers:
            raise ValueError(f"RNA modality missing required layers: {missing_layers}")
            
        # Validate RNA data quality
        self._validate_rna_quality(rna_data)
        
        # Check spatial coordinates
        required_coords = ['x_pixel', 'y_pixel']
        missing_coords = [coord for coord in required_coords 
                         if coord not in adata.obs.columns]
        if missing_coords:
            warnings.warn(f"Missing spatial coordinates: {missing_coords}. "
                         "Spatial graph construction will be disabled.")
            self._has_spatial = False
        else:
            self._has_spatial = True
            self._validate_spatial_coordinates(adata.obs)
            
        # Check ATAC modality (optional)
        if 'atac' in adata.mod:
            self._has_atac = True
            self._validate_atac_modality(adata['atac'])
            
            # Check if open_chromatin layer exists or needs to be created
            if 'open_chromatin' not in rna_data.layers:
                if self.config.regulatory.use_atac_masking:
                    warnings.warn("'open_chromatin' layer not found in RNA data. "
                                 "Will be created from ATAC data.")
        else:
            self._has_atac = False
            if self.config.regulatory.use_atac_masking:
                raise ValueError("ATAC modality required for regulatory masking, "
                               "but not found in MuData object")
                               
        self._is_validated = True
        
    def preprocess_atac(self, adata_atac: ad.AnnData, 
                       scale_factor: float = 1e4,
                       n_components: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess ATAC-seq data with TF-IDF normalization and LSI.
        
        Parameters
        ----------
        adata_atac : ad.AnnData
            ATAC-seq data object.
        scale_factor : float, default 1e4
            Scaling factor for TF-IDF normalization.
        n_components : int, default 50
            Number of LSI components to compute.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            TF-IDF normalized matrix and LSI embeddings.
        """
        # TF-IDF normalization (adapted from MultiVelo)
        X = adata_atac.X.copy()
        
        # Ensure we have a sparse matrix
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)
            
        # Term frequency (TF): normalize by total counts per cell
        npeaks = np.array(X.sum(1)).flatten()
        npeaks_inv = 1.0 / np.maximum(npeaks, 1e-10)  # Avoid division by zero
        tf = X.multiply(sparse.csr_matrix(npeaks_inv).T)
        
        # Inverse document frequency (IDF): log of inverse peak frequency
        npeaks_per_peak = np.array(X.sum(0)).flatten()
        idf = np.log1p(adata_atac.n_obs / np.maximum(npeaks_per_peak, 1))
        idf_matrix = sparse.diags(idf)
        
        # Apply TF-IDF transformation
        tfidf_matrix = tf.dot(idf_matrix) * scale_factor
        
        # LSI (Latent Semantic Indexing) using SVD
        try:
            # Use scipy's sparse SVD for efficiency
            u, s, vt = svds(tfidf_matrix, k=min(n_components, 
                                               min(tfidf_matrix.shape) - 1))
            
            # Sort by singular values (descending)
            idx = np.argsort(s)[::-1]
            lsi_embeddings = u[:, idx] * s[idx]
            
        except Exception as e:
            warnings.warn(f"LSI computation failed: {e}. Using zero embeddings.")
            lsi_embeddings = np.zeros((adata_atac.n_obs, n_components))
            
        return tfidf_matrix.toarray(), lsi_embeddings
    
    def create_open_chromatin_mask(self, adata: mu.MuData) -> torch.Tensor:
        """
        Create open chromatin accessibility mask for regulatory network.
        
        Parameters
        ----------
        adata : mu.MuData
            Multi-modal data with RNA and ATAC modalities.
            
        Returns
        -------
        torch.Tensor
            Binary mask indicating open chromatin regions per gene.
            Shape: (n_cells, n_genes)
        """
        if not self._has_atac:
            raise ValueError("ATAC modality not available for chromatin mask creation")
            
        rna_data = adata['rna']
        atac_data = adata['atac']
        
        # Check if open_chromatin layer already exists
        if 'open_chromatin' in rna_data.layers:
            return torch.from_numpy(rna_data.layers['open_chromatin'].toarray().astype(np.float32))
            
        # Preprocess ATAC data
        tfidf_matrix, _ = self.preprocess_atac(atac_data)
        
        # Create gene-level accessibility scores
        # This is a simplified version - in practice, you'd want peak-to-gene mapping
        n_genes = rna_data.n_vars
        n_cells = rna_data.n_obs
        
        # For now, create a random mapping as placeholder
        # In production, use actual peak-to-gene annotations
        np.random.seed(42)  # For reproducibility
        peak_to_gene_map = np.random.randint(0, n_genes, size=atac_data.n_vars)
        
        # Aggregate ATAC accessibility by gene
        gene_accessibility = np.zeros((n_cells, n_genes))
        for peak_idx, gene_idx in enumerate(peak_to_gene_map):
            gene_accessibility[:, gene_idx] += tfidf_matrix[:, peak_idx]
            
        # Apply threshold to create binary mask
        threshold = self.config.regulatory.atac_threshold
        open_chromatin_mask = (gene_accessibility > threshold).astype(np.float32)
        
        # Store in RNA data for future use
        rna_data.layers['open_chromatin'] = sparse.csr_matrix(open_chromatin_mask)
        
        return torch.from_numpy(open_chromatin_mask)
    
    def _validate_rna_quality(self, rna_data: ad.AnnData) -> None:
        """Validate RNA data quality."""
        # Check for empty cells/genes
        if rna_data.n_obs < 10:
            raise ValueError(f"Too few cells: {rna_data.n_obs} (minimum 10)")
            
        if rna_data.n_vars < 100:
            raise ValueError(f"Too few genes: {rna_data.n_vars} (minimum 100)")
            
        # Check spliced/unspliced data quality
        for layer_name in ['spliced', 'unspliced']:
            layer_data = rna_data.layers[layer_name]
            
            if sparse.issparse(layer_data):
                total_counts = np.array(layer_data.sum())
            else:
                total_counts = np.sum(layer_data)
                
            if total_counts == 0:
                raise ValueError(f"Layer '{layer_name}' contains no counts")
                
    def _validate_spatial_coordinates(self, obs: pd.DataFrame) -> None:
        """Validate spatial coordinate data."""
        for coord in ['x_pixel', 'y_pixel']:
            if obs[coord].isna().any():
                raise ValueError(f"Spatial coordinate '{coord}' contains NaN values")
                
            if not np.isfinite(obs[coord]).all():
                raise ValueError(f"Spatial coordinate '{coord}' contains infinite values")
                
    def _validate_atac_modality(self, atac_data: ad.AnnData) -> None:
        """Validate ATAC modality data."""
        if atac_data.n_obs == 0 or atac_data.n_vars == 0:
            raise ValueError("ATAC modality is empty")
            
        # Check for reasonable peak counts
        if sparse.issparse(atac_data.X):
            total_counts = atac_data.X.sum()
        else:
            total_counts = np.sum(atac_data.X)
            
        if total_counts == 0:
            raise ValueError("ATAC modality contains no counts")
            
    def _combine_graphs(self, spatial_graph, expression_graph):
        """Combine spatial and expression graphs for node2vec embedding."""
        # This is a placeholder implementation
        # In practice, you'd want a more sophisticated graph combination strategy
        return spatial_graph  # Use spatial graph as primary