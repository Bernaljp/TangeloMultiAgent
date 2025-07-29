"""
Tests for MuDataProcessor functionality.

This module tests multi-modal data processing including validation,
ATAC preprocessing, and integration with the graph construction pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from scipy import sparse
import muon as mu
import anndata as ad

from tangelo_velocity.config import TangeloConfig, get_stage_config
from tangelo_velocity.preprocessing import MuDataProcessor


class TestMuDataProcessor:
    """Test suite for MuDataProcessor class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return get_stage_config(0)  # Stage 0 configuration
    
    @pytest.fixture
    def sample_mudata(self):
        """Create synthetic MuData object for testing."""
        n_cells = 100
        n_rna_genes = 500
        n_atac_peaks = 200
        
        # Create RNA data
        rna_spliced = sparse.random(n_cells, n_rna_genes, density=0.1, format='csr')
        rna_unspliced = sparse.random(n_cells, n_rna_genes, density=0.05, format='csr') 
        
        rna_adata = ad.AnnData(X=rna_spliced)
        rna_adata.layers['spliced'] = rna_spliced
        rna_adata.layers['unspliced'] = rna_unspliced
        rna_adata.var_names = [f'Gene_{i}' for i in range(n_rna_genes)]
        rna_adata.obs_names = [f'Cell_{i}' for i in range(n_cells)]
        
        # Create ATAC data
        atac_data = sparse.random(n_cells, n_atac_peaks, density=0.03, format='csr')
        atac_adata = ad.AnnData(X=atac_data)
        atac_adata.var_names = [f'Peak_{i}' for i in range(n_atac_peaks)]
        atac_adata.obs_names = [f'Cell_{i}' for i in range(n_cells)]
        
        # Create MuData with spatial coordinates
        mdata = mu.MuData({'rna': rna_adata, 'atac': atac_adata})
        mdata.obs['x_pixel'] = np.random.uniform(0, 1000, n_cells)
        mdata.obs['y_pixel'] = np.random.uniform(0, 1000, n_cells)
        
        return mdata
    
    @pytest.fixture
    def minimal_mudata(self):
        """Create minimal valid MuData object."""
        n_cells = 20
        n_genes = 100
        
        # Minimal RNA data
        rna_spliced = sparse.random(n_cells, n_genes, density=0.1, format='csr')
        rna_unspliced = sparse.random(n_cells, n_genes, density=0.05, format='csr')
        
        rna_adata = ad.AnnData(X=rna_spliced)
        rna_adata.layers['spliced'] = rna_spliced
        rna_adata.layers['unspliced'] = rna_unspliced
        
        mdata = mu.MuData({'rna': rna_adata})
        mdata.obs['x_pixel'] = np.arange(n_cells, dtype=float)
        mdata.obs['y_pixel'] = np.arange(n_cells, dtype=float)
        
        return mdata
    
    def test_initialization(self, config):
        """Test MuDataProcessor initialization."""
        processor = MuDataProcessor(config)
        
        assert processor.config == config
        assert hasattr(processor, 'graph_builder')
        assert not processor._is_validated
        assert not processor._has_atac
        assert not processor._has_spatial
    
    def test_validate_mudata_success(self, config, sample_mudata):
        """Test successful MuData validation."""
        processor = MuDataProcessor(config)
        
        # Should not raise any exceptions
        processor.validate_mudata(sample_mudata)
        
        assert processor._is_validated
        assert processor._has_atac
        assert processor._has_spatial
    
    def test_validate_mudata_missing_rna(self, config):
        """Test validation failure with missing RNA modality."""
        processor = MuDataProcessor(config)
        
        # Create MuData without RNA
        mdata = mu.MuData({'atac': ad.AnnData(X=sparse.random(10, 100, format='csr'))})
        
        with pytest.raises(ValueError, match="must contain 'rna' modality"):
            processor.validate_mudata(mdata)
    
    def test_validate_mudata_missing_layers(self, config):
        """Test validation failure with missing required layers."""
        processor = MuDataProcessor(config)
        
        # Create RNA data without required layers
        rna_adata = ad.AnnData(X=sparse.random(10, 100, format='csr'))
        mdata = mu.MuData({'rna': rna_adata})
        
        with pytest.raises(ValueError, match="missing required layers"):
            processor.validate_mudata(mdata)
    
    def test_validate_mudata_missing_spatial(self, config, minimal_mudata):
        """Test validation with missing spatial coordinates."""
        processor = MuDataProcessor(config)
        
        # Remove spatial coordinates
        del minimal_mudata.obs['x_pixel']
        
        with pytest.warns(UserWarning, match="Missing spatial coordinates"):
            processor.validate_mudata(minimal_mudata)
        
        assert not processor._has_spatial
    
    def test_validate_mudata_too_few_cells(self, config):
        """Test validation failure with insufficient cells."""
        processor = MuDataProcessor(config)
        
        # Create data with too few cells
        n_cells = 5  # Below minimum of 10
        rna_adata = ad.AnnData(X=sparse.random(n_cells, 100, format='csr'))
        rna_adata.layers['spliced'] = sparse.random(n_cells, 100, format='csr')
        rna_adata.layers['unspliced'] = sparse.random(n_cells, 100, format='csr')
        
        mdata = mu.MuData({'rna': rna_adata})
        
        with pytest.raises(ValueError, match="Too few cells"):
            processor.validate_mudata(mdata)
    
    def test_preprocess_atac_tfidf(self, config, sample_mudata):
        """Test ATAC TF-IDF preprocessing."""
        processor = MuDataProcessor(config)
        atac_data = sample_mudata['atac']
        
        tfidf_matrix, lsi_embeddings = processor.preprocess_atac(atac_data)
        
        # Check output shapes
        assert tfidf_matrix.shape == atac_data.X.shape
        assert lsi_embeddings.shape[0] == atac_data.n_obs
        assert lsi_embeddings.shape[1] <= 50  # Default n_components
        
        # Check that TF-IDF normalization was applied
        assert np.all(tfidf_matrix >= 0)
        assert not np.array_equal(tfidf_matrix, atac_data.X.toarray())
    
    def test_preprocess_atac_empty_data(self, config):
        """Test ATAC preprocessing with empty data."""
        processor = MuDataProcessor(config)
        
        # Create empty ATAC data
        atac_data = ad.AnnData(X=sparse.csr_matrix((10, 50)))
        
        tfidf_matrix, lsi_embeddings = processor.preprocess_atac(atac_data)
        
        # Should handle empty data gracefully
        assert tfidf_matrix.shape == (10, 50)
        assert lsi_embeddings.shape == (10, 50)
    
    def test_create_open_chromatin_mask(self, config, sample_mudata):
        """Test open chromatin mask creation."""
        processor = MuDataProcessor(config)
        processor.validate_mudata(sample_mudata)
        
        mask = processor.create_open_chromatin_mask(sample_mudata)
        
        # Check output properties
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (sample_mudata.n_obs, sample_mudata['rna'].n_vars)
        assert mask.dtype == torch.float32
        assert torch.all((mask == 0) | (mask == 1))  # Binary mask
    
    def test_create_open_chromatin_mask_no_atac(self, config, minimal_mudata):
        """Test mask creation failure without ATAC data."""
        processor = MuDataProcessor(config)
        processor.validate_mudata(minimal_mudata)
        
        with pytest.raises(ValueError, match="ATAC modality not available"):
            processor.create_open_chromatin_mask(minimal_mudata)
    
    def test_process_mudata_complete(self, config, sample_mudata):
        """Test complete MuData processing pipeline."""
        processor = MuDataProcessor(config)
        
        result = processor.process_mudata(sample_mudata)
        
        # Check required outputs
        assert 'spliced' in result
        assert 'unspliced' in result
        assert 'spatial_graph' in result
        assert 'expression_graph' in result
        
        # Check tensor properties
        assert isinstance(result['spliced'], torch.Tensor)
        assert isinstance(result['unspliced'], torch.Tensor)
        assert result['spliced'].dtype == torch.float32
        assert result['unspliced'].dtype == torch.float32
        
        # Check shapes
        n_cells, n_genes = sample_mudata['rna'].shape
        assert result['spliced'].shape == (n_cells, n_genes)
        assert result['unspliced'].shape == (n_cells, n_genes)
    
    def test_process_mudata_with_atac_masking(self, config, sample_mudata):
        """Test processing with ATAC masking enabled."""
        # Enable ATAC masking
        config.regulatory.use_atac_masking = True
        processor = MuDataProcessor(config)
        
        result = processor.process_mudata(sample_mudata)
        
        # Should include ATAC mask
        assert 'atac_mask' in result
        assert isinstance(result['atac_mask'], torch.Tensor)
    
    def test_process_mudata_with_node2vec(self, config, sample_mudata):
        """Test processing with Node2Vec enabled."""
        # Enable Node2Vec
        config.graph.use_node2vec = True
        processor = MuDataProcessor(config)
        
        result = processor.process_mudata(sample_mudata)
        
        # Should include embeddings
        assert 'node2vec_embeddings' in result
        assert isinstance(result['node2vec_embeddings'], torch.Tensor)
    
    def test_rna_quality_validation(self, config):
        """Test RNA data quality validation."""
        processor = MuDataProcessor(config)
        
        # Test with too few genes
        rna_data = ad.AnnData(X=sparse.random(100, 50, format='csr'))  # 50 < 100 minimum
        rna_data.layers['spliced'] = sparse.random(100, 50, format='csr')
        rna_data.layers['unspliced'] = sparse.random(100, 50, format='csr')
        
        with pytest.raises(ValueError, match="Too few genes"):
            processor._validate_rna_quality(rna_data)
        
        # Test with empty layers
        rna_data = ad.AnnData(X=sparse.csr_matrix((100, 200)))
        rna_data.layers['spliced'] = sparse.csr_matrix((100, 200))
        rna_data.layers['unspliced'] = sparse.csr_matrix((100, 200))
        
        with pytest.raises(ValueError, match="contains no counts"):
            processor._validate_rna_quality(rna_data)
    
    def test_spatial_coordinates_validation(self, config):
        """Test spatial coordinates validation."""
        processor = MuDataProcessor(config)
        
        # Test with NaN coordinates
        obs_data = pd.DataFrame({
            'x_pixel': [1.0, 2.0, np.nan],
            'y_pixel': [1.0, 2.0, 3.0]
        })
        
        with pytest.raises(ValueError, match="contains NaN values"):
            processor._validate_spatial_coordinates(obs_data)
        
        # Test with infinite coordinates
        obs_data = pd.DataFrame({
            'x_pixel': [1.0, 2.0, np.inf],
            'y_pixel': [1.0, 2.0, 3.0]
        })
        
        with pytest.raises(ValueError, match="contains infinite values"):
            processor._validate_spatial_coordinates(obs_data)
    
    def test_memory_efficiency(self, config):
        """Test memory efficiency with large datasets."""
        # Create larger synthetic dataset
        n_cells = 1000
        n_genes = 2000
        
        rna_spliced = sparse.random(n_cells, n_genes, density=0.05, format='csr')
        rna_unspliced = sparse.random(n_cells, n_genes, density=0.02, format='csr')
        
        rna_adata = ad.AnnData(X=rna_spliced)
        rna_adata.layers['spliced'] = rna_spliced
        rna_adata.layers['unspliced'] = rna_unspliced
        
        mdata = mu.MuData({'rna': rna_adata})
        mdata.obs['x_pixel'] = np.random.uniform(0, 1000, n_cells)
        mdata.obs['y_pixel'] = np.random.uniform(0, 1000, n_cells)
        
        processor = MuDataProcessor(config)
        
        # Should process without memory errors
        result = processor.process_mudata(mdata)
        
        assert 'spliced' in result
        assert 'unspliced' in result
    
    def test_config_integration(self):
        """Test integration with different configuration stages."""
        for stage in [0, 1, 2, 3]:
            config = get_stage_config(stage)
            processor = MuDataProcessor(config)
            
            assert processor.config.development_stage == stage
            assert hasattr(processor, 'graph_builder')


@pytest.mark.parametrize("n_cells,n_genes", [
    (50, 200),
    (200, 1000),
    (500, 2000),
])
def test_scalability(n_cells, n_genes):
    """Test processor scalability with different data sizes."""
    config = get_stage_config(0)
    processor = MuDataProcessor(config)
    
    # Create synthetic data
    rna_adata = ad.AnnData(X=sparse.random(n_cells, n_genes, density=0.1, format='csr'))
    rna_adata.layers['spliced'] = sparse.random(n_cells, n_genes, density=0.1, format='csr')
    rna_adata.layers['unspliced'] = sparse.random(n_cells, n_genes, density=0.05, format='csr')
    
    mdata = mu.MuData({'rna': rna_adata})
    mdata.obs['x_pixel'] = np.random.uniform(0, 1000, n_cells)
    mdata.obs['y_pixel'] = np.random.uniform(0, 1000, n_cells)
    
    # Should process successfully
    result = processor.process_mudata(mdata)
    
    assert result['spliced'].shape == (n_cells, n_genes)
    assert result['unspliced'].shape == (n_cells, n_genes)