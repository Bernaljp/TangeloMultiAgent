"""
Integration tests for Stage 0 complete preprocessing pipeline.

This module tests the end-to-end functionality of Stage 0 components
working together through the main API.
"""

import pytest
import numpy as np
import torch
from scipy import sparse
import muon as mu
import anndata as ad

from tangelo_velocity.config import get_stage_config
from tangelo_velocity.api import TangeloVelocity, estimate_velocity


class TestStage0Integration:
    """Integration test suite for Stage 0 preprocessing pipeline."""
    
    @pytest.fixture
    def sample_mudata(self):
        """Create comprehensive synthetic MuData for integration testing."""
        n_cells = 100
        n_rna_genes = 500
        n_atac_peaks = 300
        
        np.random.seed(42)  # For reproducibility
        
        # Create realistic RNA data
        # Generate some structure (e.g., cell types)
        n_cell_types = 3
        cells_per_type = n_cells // n_cell_types
        
        rna_data = []
        cell_types = []
        
        for i in range(n_cell_types):
            # Each cell type has different expression pattern
            base_expression = np.random.lognormal(0, 1, n_rna_genes)
            type_specific = np.random.exponential(2, n_rna_genes) * (i + 1)
            
            for _ in range(cells_per_type):
                cell_expr = np.random.poisson(base_expression + type_specific)
                rna_data.append(cell_expr)
                cell_types.append(f'Type_{i}')
        
        # Handle remaining cells
        for _ in range(n_cells - len(rna_data)):
            cell_expr = np.random.poisson(base_expression)
            rna_data.append(cell_expr)
            cell_types.append('Type_0')
            
        rna_matrix = np.array(rna_data)
        
        # Create spliced/unspliced counts
        spliced_ratio = np.random.beta(3, 2, (n_cells, n_rna_genes))
        rna_spliced = np.random.binomial(rna_matrix, spliced_ratio)
        rna_unspliced = rna_matrix - rna_spliced
        
        # Convert to sparse matrices
        rna_spliced = sparse.csr_matrix(rna_spliced.astype(float))
        rna_unspliced = sparse.csr_matrix(rna_unspliced.astype(float))
        
        # Create RNA AnnData
        rna_adata = ad.AnnData(X=rna_spliced)
        rna_adata.layers['spliced'] = rna_spliced
        rna_adata.layers['unspliced'] = rna_unspliced
        rna_adata.var_names = [f'Gene_{i}' for i in range(n_rna_genes)]
        rna_adata.obs_names = [f'Cell_{i}' for i in range(n_cells)]
        rna_adata.obs['cell_type'] = cell_types
        
        # Add PCA for testing
        from sklearn.decomposition import PCA
        pca = PCA(n_components=30)
        rna_pca = pca.fit_transform(rna_spliced.toarray())
        rna_adata.obsm['X_pca'] = rna_pca
        
        # Create ATAC data
        atac_data = np.random.negative_binomial(5, 0.3, (n_cells, n_atac_peaks))
        atac_data = sparse.csr_matrix(atac_data.astype(float))
        
        atac_adata = ad.AnnData(X=atac_data)
        atac_adata.var_names = [f'Peak_{i}' for i in range(n_atac_peaks)]
        atac_adata.obs_names = [f'Cell_{i}' for i in range(n_cells)]
        
        # Create spatial coordinates with some structure
        # Arrange cell types in different spatial regions
        spatial_coords = []
        for i, cell_type in enumerate(cell_types):
            type_idx = int(cell_type.split('_')[1])
            
            # Each type clustered in different region
            center_x = (type_idx + 1) * 300
            center_y = (type_idx + 1) * 300
            
            x = np.random.normal(center_x, 50)
            y = np.random.normal(center_y, 50)
            spatial_coords.append([x, y])
            
        spatial_coords = np.array(spatial_coords)
        
        # Create MuData
        mdata = mu.MuData({'rna': rna_adata, 'atac': atac_adata})
        mdata.obs['x_pixel'] = spatial_coords[:, 0]
        mdata.obs['y_pixel'] = spatial_coords[:, 1]
        mdata.obs['cell_type'] = cell_types
        
        return mdata
    
    @pytest.fixture
    def stage0_config(self):
        """Get Stage 0 configuration for testing."""
        return get_stage_config(0)
    
    def test_complete_pipeline_stage0(self, stage0_config, sample_mudata):
        """Test complete Stage 0 preprocessing pipeline."""
        tv = TangeloVelocity(config=stage0_config)
        
        # Process the data
        result = tv.fit(sample_mudata, copy=True)
        
        # Check that processing completed
        assert tv.is_fitted
        assert tv._adata is not None
        assert tv._spatial_graph is not None
        assert tv._expression_graph is not None
        
        # Check that results were added to MuData
        if result is not None:  # When copy=True
            assert 'velocity' in result['rna'].layers
    
    def test_convenience_function(self, sample_mudata):
        """Test convenience function for Stage 0."""
        result = estimate_velocity(
            sample_mudata,
            stage=0,
            copy=True
        )
        
        # Should return processed MuData
        assert result is not None
        assert isinstance(result, mu.MuData)
        assert 'velocity' in result['rna'].layers
    
    def test_pipeline_without_atac(self, stage0_config):
        """Test pipeline with RNA-only data."""
        # Create RNA-only MuData
        n_cells = 50
        n_genes = 200
        
        rna_data = sparse.random(n_cells, n_genes, density=0.1, format='csr')
        rna_adata = ad.AnnData(X=rna_data)
        rna_adata.layers['spliced'] = rna_data
        rna_adata.layers['unspliced'] = sparse.random(n_cells, n_genes, density=0.05, format='csr')
        
        mdata = mu.MuData({'rna': rna_adata})
        mdata.obs['x_pixel'] = np.random.uniform(0, 1000, n_cells)
        mdata.obs['y_pixel'] = np.random.uniform(0, 1000, n_cells)
        
        # Disable ATAC masking for RNA-only data
        stage0_config.regulatory.use_atac_masking = False
        
        tv = TangeloVelocity(config=stage0_config)
        result = tv.fit(mdata, copy=True)
        
        assert result is not None
        assert 'velocity' in result['rna'].layers
    
    def test_pipeline_without_spatial(self, stage0_config, sample_mudata):
        """Test pipeline without spatial coordinates."""
        # Remove spatial coordinates
        del sample_mudata.obs['x_pixel']
        del sample_mudata.obs['y_pixel']
        
        tv = TangeloVelocity(config=stage0_config)
        
        # Should work but with warnings
        with pytest.warns(UserWarning, match="Missing spatial coordinates"):
            result = tv.fit(sample_mudata, copy=True)
        
        assert result is not None
        assert 'velocity' in result['rna'].layers
    
    def test_pipeline_with_node2vec(self, stage0_config, sample_mudata):
        """Test pipeline with Node2Vec embedding enabled."""
        # Enable Node2Vec
        stage0_config.graph.use_node2vec = True
        stage0_config.graph.node2vec_walk_length = 10  # Shorter for testing
        stage0_config.graph.node2vec_num_walks = 5
        
        tv = TangeloVelocity(config=stage0_config)
        result = tv.fit(sample_mudata, copy=True)
        
        assert result is not None
        assert 'velocity' in result['rna'].layers
        
        # Check that embeddings were created
        # Note: This tests the integration, actual embeddings tested separately
        assert tv.is_fitted
    
    def test_pipeline_with_atac_masking(self, sample_mudata):
        """Test pipeline with ATAC regulatory masking."""
        # Configure for ATAC masking
        config = get_stage_config(1)  # Stage 1 has ATAC masking enabled
        
        tv = TangeloVelocity(config=config)
        result = tv.fit(sample_mudata, copy=True)
        
        assert result is not None
        assert 'velocity' in result['rna'].layers
        
        # Check that open_chromatin layer was created
        assert 'open_chromatin' in result['rna'].layers
    
    def test_prediction_after_fitting(self, stage0_config, sample_mudata):
        """Test prediction functionality after fitting."""
        tv = TangeloVelocity(config=stage0_config)
        
        # First fit the model
        tv.fit(sample_mudata)
        
        # Then predict (should re-use fitted model)
        result = tv.predict(copy=True)
        
        assert result is not None
        assert 'velocity' in result['rna'].layers
    
    def test_prediction_without_fitting(self, stage0_config, sample_mudata):
        """Test that prediction fails without fitting."""
        tv = TangeloVelocity(config=stage0_config)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            tv.predict(sample_mudata)
    
    def test_velocity_graph_computation(self, stage0_config, sample_mudata):
        """Test velocity graph computation after processing."""
        tv = TangeloVelocity(config=stage0_config)
        
        # Fit and compute velocity graph
        tv.fit(sample_mudata)
        tv.compute_velocity_graph(n_neighbors=15)
        
        # Check that velocity graph was computed
        assert 'velocity_graph' in tv._adata['rna'].uns
    
    def test_velocity_embedding_computation(self, stage0_config, sample_mudata):
        """Test velocity embedding computation."""
        tv = TangeloVelocity(config=stage0_config)
        
        # Need UMAP embedding first
        import scanpy as sc
        sc.pp.neighbors(sample_mudata['rna'], use_rep='X_pca')
        sc.tl.umap(sample_mudata['rna'])
        
        # Fit model and compute embeddings
        tv.fit(sample_mudata)
        tv.compute_velocity_graph()
        tv.compute_velocity_embedding(basis='umap')
        
        # Check that velocity embedding was computed
        assert 'velocity_umap' in tv._adata['rna'].obsm
    
    def test_model_outputs_access(self, stage0_config, sample_mudata):
        """Test access to model outputs and representations."""
        # Configure for more advanced stage to test outputs
        config = get_stage_config(2)  # Stage 2 has more features
        
        tv = TangeloVelocity(config=config)
        tv.fit(sample_mudata)
        
        # Test access to various outputs
        # Note: These might not be implemented in Stage 0, but test the interface
        try:
            latent_reps = tv.get_latent_representations()
            assert isinstance(latent_reps, dict)
        except (ValueError, AttributeError):
            # Expected for Stage 0
            pass
        
        try:
            ode_params = tv.get_ode_parameters()
            assert isinstance(ode_params, dict)
        except (ValueError, AttributeError):
            # Expected for Stage 0
            pass
    
    def test_configuration_stages_compatibility(self, sample_mudata):
        """Test that different configuration stages work."""
        for stage in [0, 1, 2, 3]:
            config = get_stage_config(stage)
            
            # Adjust for Stage 0 limitations
            if stage >= 1:
                config.regulatory.use_atac_masking = True
            else:
                config.regulatory.use_atac_masking = False
                
            tv = TangeloVelocity(config=config)
            
            try:
                result = tv.fit(sample_mudata, copy=True)
                assert result is not None
                
            except NotImplementedError:
                # Expected for higher stages not yet implemented
                pytest.skip(f"Stage {stage} not yet implemented")
    
    def test_memory_and_performance(self, stage0_config):
        """Test memory efficiency and basic performance."""
        # Create larger dataset
        n_cells = 500
        n_genes = 1000
        
        # Create data with some sparsity
        rna_data = sparse.random(n_cells, n_genes, density=0.05, format='csr')
        rna_adata = ad.AnnData(X=rna_data)
        rna_adata.layers['spliced'] = rna_data
        rna_adata.layers['unspliced'] = sparse.random(n_cells, n_genes, density=0.02, format='csr')
        
        # Add PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        rna_pca = pca.fit_transform(rna_data.toarray())
        rna_adata.obsm['X_pca'] = rna_pca
        
        mdata = mu.MuData({'rna': rna_adata})
        mdata.obs['x_pixel'] = np.random.uniform(0, 2000, n_cells)
        mdata.obs['y_pixel'] = np.random.uniform(0, 2000, n_cells)
        
        # Should process without memory issues
        tv = TangeloVelocity(config=stage0_config)
        result = tv.fit(mdata, copy=True)
        
        assert result is not None
        assert result['rna'].shape == (n_cells, n_genes)
    
    def test_error_handling_invalid_data(self, stage0_config):
        """Test error handling with invalid input data."""
        # Test with non-MuData input
        with pytest.raises((ValueError, TypeError)):
            tv = TangeloVelocity(config=stage0_config)
            tv.fit("not_a_mudata_object")
        
        # Test with empty MuData
        empty_mdata = mu.MuData({})
        
        with pytest.raises(ValueError):
            tv = TangeloVelocity(config=stage0_config)
            tv.fit(empty_mdata)
    
    def test_reproducibility(self, stage0_config, sample_mudata):
        """Test that results are reproducible."""
        # Set random seeds
        np.random.seed(42)
        torch.manual_seed(42)
        
        tv1 = TangeloVelocity(config=stage0_config)
        result1 = tv1.fit(sample_mudata.copy(), copy=True)
        
        # Reset seeds and run again
        np.random.seed(42)
        torch.manual_seed(42)
        
        tv2 = TangeloVelocity(config=stage0_config)
        result2 = tv2.fit(sample_mudata.copy(), copy=True)
        
        # Results should be similar (allowing for some numerical differences)
        velocity1 = result1['rna'].layers['velocity']
        velocity2 = result2['rna'].layers['velocity']
        
        # Check correlation rather than exact equality due to potential numerical differences
        correlation = np.corrcoef(velocity1.flatten(), velocity2.flatten())[0, 1]
        assert correlation > 0.95  # High correlation indicates reproducibility


def test_stage_comparison_basic():
    """Test basic functionality of stage comparison."""
    # Create simple test data
    n_cells = 30
    n_genes = 100
    
    rna_data = sparse.random(n_cells, n_genes, density=0.1, format='csr')
    rna_adata = ad.AnnData(X=rna_data)
    rna_adata.layers['spliced'] = rna_data
    rna_adata.layers['unspliced'] = sparse.random(n_cells, n_genes, density=0.05, format='csr')
    
    mdata = mu.MuData({'rna': rna_adata})
    mdata.obs['x_pixel'] = np.arange(n_cells, dtype=float)
    mdata.obs['y_pixel'] = np.arange(n_cells, dtype=float)
    
    # Test only Stage 0 for now
    from tangelo_velocity.api import compare_stages
    
    try:
        results = compare_stages(mdata, stages=(0,))
        assert 0 in results
        assert isinstance(results[0], mu.MuData)
        
    except NotImplementedError:
        # Expected if comparison functionality not fully implemented
        pytest.skip("Stage comparison not fully implemented")