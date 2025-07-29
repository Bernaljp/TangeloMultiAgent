"""
Tests for GraphBuilder functionality.

This module tests spatial and expression graph construction,
including k-NN algorithms, distance metrics, and edge case handling.
"""

import pytest
import numpy as np
import torch
from scipy import sparse
import muon as mu
import anndata as ad

try:
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

from tangelo_velocity.config import GraphConfig, get_stage_config
from tangelo_velocity.preprocessing import GraphBuilder


@pytest.mark.skipif(not HAS_TORCH_GEOMETRIC, reason="torch_geometric not available")
class TestGraphBuilder:
    """Test suite for GraphBuilder class."""
    
    @pytest.fixture
    def config(self):
        """Create test graph configuration."""
        return GraphConfig(
            n_neighbors_spatial=5,
            n_neighbors_expression=10,
            spatial_method="knn",
            expression_metric="cosine",
            spatial_radius=100.0
        )
    
    @pytest.fixture
    def sample_coordinates(self):
        """Create sample spatial coordinates."""
        n_cells = 50
        np.random.seed(42)
        coordinates = np.random.uniform(0, 1000, (n_cells, 2))
        return coordinates.astype(np.float32)
    
    @pytest.fixture
    def sample_expression(self):
        """Create sample expression features."""
        n_cells = 50
        n_features = 100
        np.random.seed(42)
        # Create features with some structure
        features = np.random.normal(0, 1, (n_cells, n_features))
        return features.astype(np.float32)
    
    @pytest.fixture
    def sample_mudata(self):
        """Create synthetic MuData for graph testing."""
        n_cells = 50
        n_genes = 200
        
        # RNA data with PCA
        rna_data = sparse.random(n_cells, n_genes, density=0.1, format='csr')
        rna_adata = ad.AnnData(X=rna_data)
        
        # Add PCA embedding
        np.random.seed(42)
        rna_adata.obsm['X_pca'] = np.random.normal(0, 1, (n_cells, 30))
        
        # Create MuData with spatial coordinates
        mdata = mu.MuData({'rna': rna_adata})
        mdata.obs['x_pixel'] = np.random.uniform(0, 1000, n_cells)
        mdata.obs['y_pixel'] = np.random.uniform(0, 1000, n_cells)
        
        return mdata
    
    def test_initialization(self, config):
        """Test GraphBuilder initialization."""
        builder = GraphBuilder(config)
        
        assert builder.config == config
        assert hasattr(builder, 'config')
    
    def test_build_spatial_graph_knn(self, config, sample_coordinates):
        """Test spatial k-NN graph construction."""
        builder = GraphBuilder(config)
        
        graph = builder.build_spatial_graph(sample_coordinates)
        
        # Check graph properties
        assert isinstance(graph, Data)
        assert graph.num_nodes == len(sample_coordinates)
        assert graph.x.shape == (len(sample_coordinates), 2)
        
        # Check edges exist
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0
        
        # Check edge attributes
        assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
        assert torch.all(graph.edge_attr >= 0)  # Non-negative weights
    
    def test_build_spatial_graph_radius(self, sample_coordinates):
        """Test spatial radius graph construction."""
        config = GraphConfig(
            spatial_method="radius",
            spatial_radius=200.0,
            n_neighbors_spatial=5
        )
        builder = GraphBuilder(config)
        
        graph = builder.build_spatial_graph(sample_coordinates)
        
        # Check graph properties
        assert isinstance(graph, Data)
        assert graph.num_nodes == len(sample_coordinates)
        
        # Should have edges within radius
        assert graph.edge_index.shape[1] > 0
    
    def test_build_expression_graph(self, config, sample_expression):
        """Test expression similarity graph construction."""
        builder = GraphBuilder(config)
        
        graph = builder.build_expression_graph(sample_expression)
        
        # Check graph properties
        assert isinstance(graph, Data)
        assert graph.num_nodes == len(sample_expression)
        assert graph.x.shape == sample_expression.shape
        
        # Check edges
        assert graph.edge_index.shape[0] == 2
        assert graph.edge_index.shape[1] > 0
        
        # Check that edges respect k-NN constraint
        max_edges_per_node = config.n_neighbors_expression * 2  # Undirected
        edges_per_node = torch.bincount(graph.edge_index[0])
        assert torch.all(edges_per_node <= max_edges_per_node)
    
    def test_build_graphs_complete(self, config, sample_mudata):
        """Test complete graph building pipeline."""
        builder = GraphBuilder(config)
        
        spatial_graph, expression_graph = builder.build_graphs(sample_mudata)
        
        # Check both graphs
        assert isinstance(spatial_graph, Data)
        assert isinstance(expression_graph, Data)
        
        # Same number of nodes
        assert spatial_graph.num_nodes == expression_graph.num_nodes
        assert spatial_graph.num_nodes == sample_mudata.n_obs
    
    def test_knn_graph_euclidean(self, config, sample_expression):
        """Test k-NN graph with Euclidean distance."""
        config.expression_metric = "euclidean"
        builder = GraphBuilder(config)
        
        adj_matrix = builder._compute_knn_graph(
            sample_expression, 
            k=5, 
            metric="euclidean"
        )
        
        # Check adjacency matrix properties
        assert adj_matrix.shape == (len(sample_expression), len(sample_expression))
        assert sparse.issparse(adj_matrix)
        
        # Check symmetry (undirected graph)
        diff = adj_matrix - adj_matrix.T
        assert np.allclose(diff.data, 0, atol=1e-6)
        
        # Check no self-loops
        assert adj_matrix.diagonal().sum() == 0
    
    def test_knn_graph_cosine(self, config, sample_expression):
        """Test k-NN graph with cosine similarity."""
        config.expression_metric = "cosine"
        builder = GraphBuilder(config)
        
        adj_matrix = builder._compute_knn_graph(
            sample_expression,
            k=5,
            metric="cosine"
        )
        
        # Check properties
        assert adj_matrix.shape == (len(sample_expression), len(sample_expression))
        assert sparse.issparse(adj_matrix)
        
        # Cosine similarities should be in [0, 1] after conversion
        assert np.all(adj_matrix.data >= 0)
        assert np.all(adj_matrix.data <= 1)
    
    def test_radius_graph_construction(self, config, sample_coordinates):
        """Test radius-based graph construction."""
        builder = GraphBuilder(config)
        
        adj_matrix = builder._compute_radius_graph(sample_coordinates, radius=200.0)
        
        # Check properties
        assert adj_matrix.shape == (len(sample_coordinates), len(sample_coordinates))
        assert sparse.issparse(adj_matrix)
        
        # Check that distances respect radius constraint
        # (This is an indirect test through the resulting adjacency matrix)
        assert adj_matrix.nnz > 0  # Should have some connections
    
    def test_sparse_to_edge_format(self, config):
        """Test conversion from sparse matrix to edge format."""
        builder = GraphBuilder(config)
        
        # Create simple test adjacency matrix
        row = np.array([0, 0, 1, 2])
        col = np.array([1, 2, 2, 0])
        data = np.array([0.5, 0.8, 0.3, 0.9])
        adj_matrix = sparse.csr_matrix((data, (row, col)), shape=(3, 3))
        
        edge_index, edge_attr = builder._sparse_to_edge_format(adj_matrix)
        
        # Check output format
        assert isinstance(edge_index, torch.Tensor)
        assert isinstance(edge_attr, torch.Tensor)
        assert edge_index.dtype == torch.long
        assert edge_attr.dtype == torch.float32
        
        # Check dimensions
        assert edge_index.shape == (2, len(data))
        assert edge_attr.shape == (len(data),)
        
        # Check values
        assert torch.allclose(edge_attr, torch.tensor(data, dtype=torch.float32))
    
    def test_extract_spatial_coordinates(self, config, sample_mudata):
        """Test spatial coordinate extraction."""
        builder = GraphBuilder(config)
        
        coords = builder._extract_spatial_coordinates(sample_mudata)
        
        # Check output
        assert isinstance(coords, np.ndarray)
        assert coords.dtype == np.float32
        assert coords.shape == (sample_mudata.n_obs, 2)
        
        # Check that coordinates match input
        expected = sample_mudata.obs[['x_pixel', 'y_pixel']].values
        assert np.allclose(coords, expected)
    
    def test_extract_expression_features_pca(self, config, sample_mudata):
        """Test expression feature extraction with PCA."""
        builder = GraphBuilder(config)
        
        features = builder._extract_expression_features(sample_mudata)
        
        # Should use PCA features
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert features.shape[0] == sample_mudata.n_obs
        assert features.shape[1] <= 50  # Limited to 50 PCs
    
    def test_extract_expression_features_raw(self, config):
        """Test expression feature extraction without PCA."""
        # Create MuData without PCA
        n_cells = 30
        n_genes = 100
        
        rna_data = sparse.random(n_cells, n_genes, density=0.1, format='csr')
        rna_adata = ad.AnnData(X=rna_data)
        mdata = mu.MuData({'rna': rna_adata})
        
        builder = GraphBuilder(config)
        features = builder._extract_expression_features(mdata)
        
        # Should use raw expression
        assert isinstance(features, np.ndarray)
        assert features.shape == (n_cells, n_genes)
    
    def test_extract_expression_features_hvg(self, config):
        """Test feature extraction with highly variable genes."""
        # Create MuData with many genes and HVG annotation
        n_cells = 30
        n_genes = 3000  # Exceeds 2000 threshold
        
        rna_data = sparse.random(n_cells, n_genes, density=0.05, format='csr')
        rna_adata = ad.AnnData(X=rna_data)
        
        # Add highly variable gene annotation
        hvg_mask = np.zeros(n_genes, dtype=bool)
        hvg_mask[:500] = True  # First 500 genes are HVG
        rna_adata.var['highly_variable'] = hvg_mask
        
        mdata = mu.MuData({'rna': rna_adata})
        
        builder = GraphBuilder(config)
        features = builder._extract_expression_features(mdata)
        
        # Should use only HVG
        assert features.shape == (n_cells, 500)
    
    def test_missing_spatial_coordinates(self, config):
        """Test handling of missing spatial coordinates."""
        # Create MuData without spatial coordinates
        n_cells = 20
        rna_adata = ad.AnnData(X=sparse.random(n_cells, 100, format='csr'))
        mdata = mu.MuData({'rna': rna_adata})
        
        builder = GraphBuilder(config)
        
        # Should not have spatial coordinates
        assert not builder._has_spatial_coordinates(mdata)
        
        # Should create empty spatial graph
        spatial_graph, expression_graph = builder.build_graphs(mdata)
        
        assert spatial_graph.edge_index.shape[1] == 0  # No edges
        assert expression_graph.edge_index.shape[1] > 0  # Has edges
    
    def test_invalid_spatial_coordinates(self, config, sample_mudata):
        """Test handling of invalid spatial coordinates."""
        builder = GraphBuilder(config)
        
        # Add NaN coordinates
        sample_mudata.obs.loc[sample_mudata.obs.index[0], 'x_pixel'] = np.nan
        
        with pytest.raises(ValueError, match="contain NaN or infinite values"):
            builder._extract_spatial_coordinates(sample_mudata)
    
    def test_edge_case_small_k(self, config):
        """Test k-NN with k larger than available neighbors."""
        # Create very small dataset
        coordinates = np.array([[0, 0], [1, 1]], dtype=np.float32)
        
        # Set k larger than available neighbors
        config.n_neighbors_spatial = 10
        builder = GraphBuilder(config)
        
        graph = builder.build_spatial_graph(coordinates)
        
        # Should handle gracefully
        assert isinstance(graph, Data)
        assert graph.num_nodes == 2
    
    def test_edge_case_identical_features(self, config):
        """Test graph construction with identical features."""
        # Create features where all cells are identical
        n_cells = 20
        features = np.ones((n_cells, 50), dtype=np.float32)
        
        builder = GraphBuilder(config)
        
        # Should handle without errors
        graph = builder.build_expression_graph(features)
        
        assert isinstance(graph, Data)
        assert graph.num_nodes == n_cells
    
    def test_different_distance_metrics(self, config, sample_expression):
        """Test various distance metrics."""
        metrics = ['euclidean', 'cosine', 'manhattan', 'chebyshev']
        builder = GraphBuilder(config)
        
        for metric in metrics:
            config.expression_metric = metric
            
            try:
                adj_matrix = builder._compute_knn_graph(
                    sample_expression, k=5, metric=metric
                )
                
                # Check basic properties
                assert sparse.issparse(adj_matrix)
                assert adj_matrix.shape[0] == len(sample_expression)
                
            except Exception as e:
                pytest.fail(f"Failed with metric {metric}: {e}")
    
    def test_graph_symmetry(self, config, sample_coordinates):
        """Test that constructed graphs are symmetric (undirected)."""
        builder = GraphBuilder(config)
        
        graph = builder.build_spatial_graph(sample_coordinates)
        
        # Convert back to adjacency matrix for symmetry check
        adj_matrix = sparse.coo_matrix(
            (graph.edge_attr.numpy(), 
             (graph.edge_index[0].numpy(), graph.edge_index[1].numpy())),
            shape=(graph.num_nodes, graph.num_nodes)
        ).tocsr()
        
        # Check symmetry
        diff = adj_matrix - adj_matrix.T
        assert np.allclose(diff.data, 0, atol=1e-6)
    
    def test_memory_efficiency(self, config):
        """Test memory efficiency with larger graphs."""
        # Create larger dataset
        n_cells = 500
        coordinates = np.random.uniform(0, 1000, (n_cells, 2)).astype(np.float32)
        
        builder = GraphBuilder(config)
        
        # Should handle without memory issues
        graph = builder.build_spatial_graph(coordinates)
        
        assert graph.num_nodes == n_cells
        assert isinstance(graph, Data)


@pytest.mark.parametrize("n_neighbors,metric", [
    (5, "euclidean"),
    (10, "cosine"), 
    (15, "manhattan"),
])
def test_parameter_combinations(n_neighbors, metric):
    """Test different parameter combinations."""
    config = GraphConfig(
        n_neighbors_expression=n_neighbors,
        expression_metric=metric
    )
    
    builder = GraphBuilder(config)
    
    # Create test data
    features = np.random.normal(0, 1, (50, 30)).astype(np.float32)
    
    graph = builder.build_expression_graph(features)
    
    assert isinstance(graph, Data)
    assert graph.num_nodes == 50


def test_empty_graph_creation():
    """Test creation of empty graphs."""
    config = GraphConfig()
    builder = GraphBuilder(config)
    
    empty_graph = builder._create_empty_graph(n_nodes=10)
    
    assert isinstance(empty_graph, Data)
    assert empty_graph.num_nodes == 10
    assert empty_graph.edge_index.shape[1] == 0
    assert empty_graph.x.shape == (10, 1)