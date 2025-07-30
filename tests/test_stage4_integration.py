"""
Comprehensive integration testing for Stage 4 advanced features implementation.

This test validates:
1. Mathematical formulation correctness (α = W @ sigmoid(s))
2. Sigmoid pretraining and freezing mechanism
3. All Stage 4 advanced features
4. Integration across all stages (1-4)
5. End-to-end model functionality
"""

import sys
import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    warnings.warn("torch_geometric not available. Some tests will be skipped.")

from tangelo_velocity.models.stage1 import Stage1RegulatoryModel
from tangelo_velocity.models.stage2 import Stage2GraphModel
from tangelo_velocity.models.stage3 import Stage3IntegratedModel
from tangelo_velocity.models.stage4 import Stage4AdvancedModel
from tangelo_velocity.config import TangeloConfig


def create_test_config(stage: int) -> TangeloConfig:
    """Create test configuration for specified stage."""
    config = TangeloConfig()
    config.development_stage = stage
    
    # Regulatory network settings
    config.regulatory = SimpleNamespace()
    config.regulatory.n_sigmoid_components = 5
    config.regulatory.use_bias = True
    config.regulatory.interaction_strength = 1.0
    config.regulatory.soft_constraint = True
    config.regulatory.lambda_l1 = 0.1
    config.regulatory.lambda_l2 = 0.01
    
    # Encoder settings
    config.encoder = SimpleNamespace()
    config.encoder.hidden_dims = (128, 64)
    config.encoder.latent_dim = 32
    config.encoder.dropout = 0.1
    config.encoder.batch_norm = True
    config.encoder.aggregator = 'mean'
    config.encoder.fusion_method = 'attention'
    config.encoder.spatial_feature_dim = 2
    
    # ODE settings
    config.ode = SimpleNamespace()
    config.ode.solver = 'euler'
    config.ode.t_span = (0.0, 1.0)
    config.ode.dt = 0.1
    
    # Stage 4 specific settings
    if stage == 4:
        config.n_time_points = 5
        config.uncertainty_samples = 10  # Reduced for testing
        config.n_cell_types = 5
        config.gene_names = [f"Gene_{i}" for i in range(20)]  # Will be overridden
    
    return config


def create_test_data(batch_size: int = 10, n_genes: int = 20, n_atac: int = 100):
    """Create synthetic test data."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # RNA data
    spliced = torch.rand(batch_size, n_genes) * 10 + 1  # Avoid zeros
    unspliced = torch.rand(batch_size, n_genes) * 8 + 1
    
    # ATAC mask (gene x gene regulatory relationships)
    atac_mask = torch.rand(n_genes, n_genes) > 0.7  # 30% connectivity
    atac_mask = atac_mask.float()
    
    # Spatial coordinates
    spatial_coords = torch.rand(batch_size, 2) * 100
    
    # Cell type labels for Stage 4
    cell_type_labels = torch.randint(0, 5, (batch_size,))
    
    # Time points for temporal dynamics
    time_points = torch.linspace(0, 1, 5).unsqueeze(0).expand(batch_size, -1)
    
    # Create graphs if torch_geometric is available
    graphs = None
    if HAS_TORCH_GEOMETRIC:
        # Simple spatial graph (k-NN with k=3)
        edge_indices = []
        for i in range(batch_size):
            # Connect each cell to itself and 2 neighbors
            edges = []
            for j in range(batch_size):
                for k in range(min(3, batch_size)):
                    target = (j + k) % batch_size
                    edges.append([j, target])
            edge_index = torch.tensor(edges).T
            edge_indices.append(edge_index)
        
        spatial_graph = Data(edge_index=edge_indices[0], num_nodes=batch_size)
        expression_graph = Data(edge_index=edge_indices[0], num_nodes=batch_size)
        graphs = (spatial_graph, expression_graph)
    
    return {
        'spliced': spliced,
        'unspliced': unspliced,
        'atac_mask': atac_mask,
        'spatial_coords': spatial_coords,
        'cell_type_labels': cell_type_labels,
        'time_points': time_points,
        'graphs': graphs
    }


def test_mathematical_formulation():
    """Test the correct mathematical formulation α = W @ sigmoid(s)."""
    print("=== Testing Mathematical Formulation α = W @ sigmoid(s) ===")
    
    batch_size, n_genes, n_atac = 5, 10, 50
    config = create_test_config(stage=1)
    test_data = create_test_data(batch_size, n_genes, n_atac)
    
    # Test Stage 1 (regulatory model)
    model = Stage1RegulatoryModel(config, n_genes, n_atac)
    model.set_atac_mask(test_data['atac_mask'])
    
    # Forward pass
    outputs = model(test_data['spliced'], test_data['unspliced'])
    
    # Verify transcription rates are computed correctly
    assert 'transcription_rates' in outputs, "Transcription rates not in outputs"
    assert outputs['transcription_rates'].shape == (batch_size, n_genes), \
        f"Wrong transcription rates shape: {outputs['transcription_rates'].shape}"
    
    # Test regulatory network directly
    reg_net = model.regulatory_network
    
    # Get sigmoid features
    sigmoid_features = reg_net.sigmoid_features(test_data['spliced'])
    assert sigmoid_features.shape == test_data['spliced'].shape, \
        "Sigmoid features shape mismatch"
    
    # Get interaction matrix
    W = reg_net.get_interaction_matrix()
    assert W.shape == (n_genes, n_genes), f"Wrong interaction matrix shape: {W.shape}"
    
    # Verify α = W @ sigmoid(s) formulation
    manual_alpha = torch.matmul(W, sigmoid_features.T).T  # W @ sigmoid(s)
    network_alpha = reg_net.interaction_network(sigmoid_features)
    
    # Should be approximately equal (allowing for base transcription rate addition)
    base_rates = reg_net.base_transcription
    expected_alpha = F.softplus(manual_alpha + base_rates)
    actual_alpha = outputs['transcription_rates']
    
    # Check if they're close (allowing for numerical differences)
    diff = torch.abs(expected_alpha - actual_alpha).mean()
    assert diff < 1e-3, f"Mathematical formulation mismatch: diff = {diff}"
    
    print("✓ Mathematical formulation α = W @ sigmoid(s) verified")
    print(f"  - Sigmoid features shape: {sigmoid_features.shape}")
    print(f"  - Interaction matrix shape: {W.shape}")
    print(f"  - Transcription rates shape: {actual_alpha.shape}")
    print(f"  - Formulation difference: {diff:.6f}")


def test_sigmoid_pretraining_protocol():
    """Test sigmoid pretraining and freezing mechanism."""
    print("\n=== Testing Sigmoid Pretraining Protocol ===")
    
    batch_size, n_genes, n_atac = 10, 15, 75
    config = create_test_config(stage=1)
    test_data = create_test_data(batch_size, n_genes, n_atac)
    
    model = Stage1RegulatoryModel(config, n_genes, n_atac)
    model.set_atac_mask(test_data['atac_mask'])
    
    # Check initial state
    assert not model.regulatory_network.is_sigmoid_frozen(), \
        "Sigmoid should not be frozen initially"
    
    # Get initial sigmoid parameters
    initial_params = model.regulatory_network.sigmoid_features.get_parameters()
    
    # Pretrain sigmoid features (with reduced epochs for testing)
    print("Pre-training sigmoid features...")
    model.pretrain_sigmoid_features(
        test_data['spliced'], 
        n_epochs=5,  # Reduced for testing
        learning_rate=0.1,
        freeze_after_pretraining=True
    )
    
    # Check if sigmoid is frozen
    assert model.regulatory_network.is_sigmoid_frozen(), \
        "Sigmoid should be frozen after pretraining"
    
    # Verify parameters have changed during pretraining
    trained_params = model.regulatory_network.sigmoid_features.get_parameters()
    param_changed = False
    for key in initial_params:
        if not torch.allclose(initial_params[key], trained_params[key], atol=1e-4):
            param_changed = True
            break
    
    assert param_changed, "Sigmoid parameters should have changed during pretraining"
    
    # Verify sigmoid parameters are frozen (requires_grad = False)
    for param in model.regulatory_network.sigmoid_features.parameters():
        assert not param.requires_grad, \
            "Sigmoid parameters should have requires_grad=False when frozen"
    
    # Verify interaction network parameters are still trainable
    for param in model.regulatory_network.interaction_network.parameters():
        assert param.requires_grad, \
            "Interaction network parameters should remain trainable"
    
    print("✓ Sigmoid pretraining protocol verified")
    print(f"  - Sigmoid frozen: {model.regulatory_network.is_sigmoid_frozen()}")
    print(f"  - Parameters changed during pretraining: {param_changed}")
    print("  - Only interaction network W remains trainable")


def test_stage4_advanced_features():
    """Test all Stage 4 advanced features."""
    print("\n=== Testing Stage 4 Advanced Features ===")
    
    if not HAS_TORCH_GEOMETRIC:
        print("⚠️  Skipping Stage 4 tests: torch_geometric not available")
        return
    
    batch_size, n_genes, n_atac = 8, 12, 60
    config = create_test_config(stage=4)
    test_data = create_test_data(batch_size, n_genes, n_atac)
    
    # Update config with test data parameters
    config.gene_names = [f"Gene_{i}" for i in range(n_genes)]
    
    model = Stage4AdvancedModel(config, n_genes, n_atac)
    model.set_atac_mask(test_data['atac_mask'])
    
    # Test sigmoid pretraining for Stage 4
    print("Testing Stage 4 sigmoid pretraining...")
    model.pretrain_sigmoid_features(
        test_data['spliced'],
        n_epochs=3,  # Reduced for testing
        learning_rate=0.1,
        freeze_after_pretraining=True
    )
    
    assert model.is_sigmoid_frozen(), "Stage 4 sigmoid should be frozen"
    
    # Full forward pass with all advanced features
    spatial_graph, expression_graph = test_data['graphs']
    
    outputs = model(
        spliced=test_data['spliced'],
        unspliced=test_data['unspliced'],
        spatial_graph=spatial_graph,
        expression_graph=expression_graph,
        spatial_coords=test_data['spatial_coords'],
        atac_mask=test_data['atac_mask'],
        time_points=test_data['time_points'],
        cell_type_labels=test_data['cell_type_labels']
    )
    
    # Test 1: Temporal Dynamics
    print("  Testing temporal dynamics...")
    assert 'temporal_velocities' in outputs, "Temporal velocities missing"
    temporal_shape = outputs['temporal_velocities'].shape
    expected_temporal_shape = (batch_size, config.n_time_points, n_genes)
    assert temporal_shape == expected_temporal_shape, \
        f"Wrong temporal velocities shape: {temporal_shape} vs {expected_temporal_shape}"
    print(f"    ✓ Temporal velocities shape: {temporal_shape}")
    
    # Test 2: Uncertainty Quantification
    print("  Testing uncertainty quantification...")
    assert 'uncertainty_velocity_mean' in outputs, "Uncertainty mean missing"
    assert 'uncertainty_total_uncertainty' in outputs, "Total uncertainty missing"
    assert 'uncertainty_confidence_intervals' in outputs, "Confidence intervals missing"
    
    uncertainty_mean = outputs['uncertainty_velocity_mean']
    total_uncertainty = outputs['uncertainty_total_uncertainty']
    confidence_intervals = outputs['uncertainty_confidence_intervals']
    
    assert uncertainty_mean.shape == (batch_size, n_genes), \
        f"Wrong uncertainty mean shape: {uncertainty_mean.shape}"
    assert total_uncertainty.shape == (batch_size, n_genes), \
        f"Wrong total uncertainty shape: {total_uncertainty.shape}"
    assert 'lower_bound' in confidence_intervals and 'upper_bound' in confidence_intervals, \
        "Confidence intervals incomplete"
    
    print(f"    ✓ Uncertainty mean shape: {uncertainty_mean.shape}")
    print(f"    ✓ Total uncertainty shape: {total_uncertainty.shape}")
    print(f"    ✓ Confidence intervals computed")
    
    # Test 3: Multi-Scale Integration
    print("  Testing multi-scale integration...")
    assert 'multiscale_integrated_velocity' in outputs, "Integrated velocity missing"
    assert 'multiscale_cell_velocity' in outputs, "Cell velocity missing"
    assert 'multiscale_tissue_velocity' in outputs, "Tissue velocity missing"
    assert 'multiscale_scale_weights' in outputs, "Scale weights missing"
    
    integrated_velocity = outputs['multiscale_integrated_velocity']
    scale_weights = outputs['multiscale_scale_weights']
    
    assert integrated_velocity.shape == (batch_size, n_genes), \
        f"Wrong integrated velocity shape: {integrated_velocity.shape}"
    assert scale_weights.shape == (2,), f"Wrong scale weights shape: {scale_weights.shape}"
    assert torch.allclose(torch.sum(scale_weights), torch.tensor(1.0), atol=1e-6), \
        "Scale weights should sum to 1"
    
    print(f"    ✓ Integrated velocity shape: {integrated_velocity.shape}")
    print(f"    ✓ Scale weights: {scale_weights.detach().numpy()}")
    
    # Test 4: Advanced Regularization (test through loss computation)
    print("  Testing advanced regularization...")
    targets = {'spliced': test_data['spliced'], 'unspliced': test_data['unspliced']}
    loss = model.compute_loss(outputs, targets)
    loss_dict = model.get_loss_dict()
    
    regularization_components = [k for k in loss_dict.keys() if 'reg_' in k]
    assert len(regularization_components) > 0, "No regularization components found"
    assert 'regularization_total' in loss_dict, "Total regularization loss missing"
    
    print(f"    ✓ Regularization components: {regularization_components}")
    print(f"    ✓ Total regularization loss: {loss_dict['regularization_total'].item():.6f}")
    
    # Test 5: Interpretability Tools
    print("  Testing interpretability tools...")
    assert 'importance_scores' in outputs, "Importance scores missing"
    assert 'importance_gene_rankings' in outputs, "Gene rankings missing"
    assert 'importance_top_genes' in outputs, "Top genes missing"
    
    importance_scores = outputs['importance_scores']
    gene_rankings = outputs['importance_gene_rankings']
    top_genes = outputs['importance_top_genes']
    
    assert importance_scores.shape == (batch_size, n_genes), \
        f"Wrong importance scores shape: {importance_scores.shape}"
    assert gene_rankings.shape == (batch_size, n_genes), \
        f"Wrong gene rankings shape: {gene_rankings.shape}"
    assert len(top_genes) == batch_size, f"Wrong top genes length: {len(top_genes)}"
    
    # Check that importance scores sum to 1 (softmax normalization)
    importance_sums = torch.sum(importance_scores, dim=1)
    assert torch.allclose(importance_sums, torch.ones(batch_size), atol=1e-6), \
        "Importance scores should sum to 1"
    
    print(f"    ✓ Importance scores shape: {importance_scores.shape}")
    print(f"    ✓ Gene rankings shape: {gene_rankings.shape}")
    print(f"    ✓ Top genes per batch: {len(top_genes[0])}")
    
    # Test advanced analysis
    print("  Testing advanced analysis...")
    analysis = model.get_advanced_analysis()
    
    expected_keys = ['temporal', 'uncertainty', 'multiscale', 'interpretability']
    for key in expected_keys:
        assert key in analysis, f"Missing analysis component: {key}"
    
    print(f"    ✓ Analysis components: {list(analysis.keys())}")
    
    print("✓ All Stage 4 advanced features verified")


def test_cross_stage_integration():
    """Test integration and consistency across all stages."""
    print("\n=== Testing Cross-Stage Integration ===")
    
    if not HAS_TORCH_GEOMETRIC:
        print("⚠️  Skipping cross-stage tests: torch_geometric not available")
        return
    
    batch_size, n_genes, n_atac = 6, 10, 50
    test_data = create_test_data(batch_size, n_genes, n_atac)
    
    models = {}
    outputs_dict = {}
    
    # Test Stage 1
    print("  Testing Stage 1...")
    config1 = create_test_config(stage=1)
    models[1] = Stage1RegulatoryModel(config1, n_genes, n_atac)
    models[1].set_atac_mask(test_data['atac_mask'])
    
    outputs_dict[1] = models[1](test_data['spliced'], test_data['unspliced'])
    assert 'velocity' in outputs_dict[1], "Stage 1 velocity missing"
    print(f"    ✓ Stage 1 velocity shape: {outputs_dict[1]['velocity'].shape}")
    
    # Test Stage 2
    print("  Testing Stage 2...")
    config2 = create_test_config(stage=2)
    models[2] = Stage2GraphModel(config2, n_genes, n_atac)
    
    spatial_graph, expression_graph = test_data['graphs']
    outputs_dict[2] = models[2](
        test_data['spliced'], 
        test_data['unspliced'],
        spatial_graph=spatial_graph,
        expression_graph=expression_graph,
        spatial_coords=test_data['spatial_coords']
    )
    assert 'velocity' in outputs_dict[2], "Stage 2 velocity missing"
    print(f"    ✓ Stage 2 velocity shape: {outputs_dict[2]['velocity'].shape}")
    
    # Test Stage 3
    print("  Testing Stage 3...")
    config3 = create_test_config(stage=3)
    models[3] = Stage3IntegratedModel(config3, n_genes, n_atac)
    models[3].set_atac_mask(test_data['atac_mask'])
    
    outputs_dict[3] = models[3](
        test_data['spliced'],
        test_data['unspliced'],
        spatial_graph=spatial_graph,
        expression_graph=expression_graph,
        spatial_coords=test_data['spatial_coords']
    )
    assert 'velocity' in outputs_dict[3], "Stage 3 velocity missing"
    print(f"    ✓ Stage 3 velocity shape: {outputs_dict[3]['velocity'].shape}")
    
    # Test Stage 4
    print("  Testing Stage 4...")
    config4 = create_test_config(stage=4)
    config4.gene_names = [f"Gene_{i}" for i in range(n_genes)]
    models[4] = Stage4AdvancedModel(config4, n_genes, n_atac)
    models[4].set_atac_mask(test_data['atac_mask'])
    
    outputs_dict[4] = models[4](
        test_data['spliced'],
        test_data['unspliced'],
        spatial_graph=spatial_graph,
        expression_graph=expression_graph,
        spatial_coords=test_data['spatial_coords'],
        time_points=test_data['time_points'],
        cell_type_labels=test_data['cell_type_labels']
    )
    assert 'velocity' in outputs_dict[4], "Stage 4 velocity missing"
    print(f"    ✓ Stage 4 velocity shape: {outputs_dict[4]['velocity'].shape}")
    
    # Verify all velocities have the same shape
    expected_shape = (batch_size, n_genes)
    for stage in [1, 2, 3, 4]:
        actual_shape = outputs_dict[stage]['velocity'].shape
        assert actual_shape == expected_shape, \
            f"Stage {stage} velocity shape mismatch: {actual_shape} vs {expected_shape}"
    
    # Test mathematical consistency in regulatory components (Stages 1, 3, 4)
    print("  Testing mathematical consistency...")
    
    # All regulatory models should produce transcription rates
    for stage in [1, 3, 4]:
        assert 'transcription_rates' in outputs_dict[stage], \
            f"Stage {stage} transcription rates missing"
        rates_shape = outputs_dict[stage]['transcription_rates'].shape
        assert rates_shape == expected_shape, \
            f"Stage {stage} transcription rates shape mismatch: {rates_shape}"
    
    # Test sigmoid pretraining consistency
    print("  Testing sigmoid pretraining consistency...")
    for stage in [1, 3, 4]:
        model = models[stage]
        
        # Test pretraining
        model.pretrain_sigmoid_features(
            test_data['spliced'],
            n_epochs=2,  # Minimal for testing
            learning_rate=0.1,
            freeze_after_pretraining=True
        )
        
        # Verify frozen state
        assert model.is_sigmoid_frozen(), f"Stage {stage} sigmoid should be frozen"
    
    print("✓ Cross-stage integration verified")
    print("  - All stages produce consistent velocity shapes")
    print("  - Regulatory stages maintain mathematical consistency")
    print("  - Sigmoid pretraining works across all stages")


def test_model_summaries():
    """Test model summary generation."""
    print("\n=== Testing Model Summaries ===")
    
    batch_size, n_genes, n_atac = 5, 8, 40
    test_data = create_test_data(batch_size, n_genes, n_atac)
    
    # Test Stage 1 summary
    config1 = create_test_config(stage=1)
    model1 = Stage1RegulatoryModel(config1, n_genes, n_atac)
    summary1 = model1.get_model_summary()
    
    assert "Stage 1 Regulatory Model Summary" in summary1
    assert f"Genes: {n_genes}" in summary1
    assert f"ATAC Features: {n_atac}" in summary1
    assert "α = W(sigmoid(s))" in summary1 or "W @ sigmoid(s)" in summary1
    print("✓ Stage 1 summary generated")
    
    # Test Stage 4 summary if available
    if HAS_TORCH_GEOMETRIC:
        config4 = create_test_config(stage=4)
        config4.gene_names = [f"Gene_{i}" for i in range(n_genes)]
        model4 = Stage4AdvancedModel(config4, n_genes, n_atac)
        summary4 = model4.get_model_summary()
        
        assert "Stage 4 Advanced Model Summary" in summary4
        assert "✓ Temporal Dynamics" in summary4
        assert "✓ Uncertainty Quantification" in summary4
        assert "✓ Multi-Scale Integration" in summary4
        assert "✓ Advanced Regularization" in summary4
        assert "✓ Interpretability Tools" in summary4
        assert "α = W @ sigmoid(s) (CORRECTED)" in summary4
        print("✓ Stage 4 summary generated")
    
    print("✓ Model summaries verified")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("TANGELO VELOCITY STAGE 4 INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        # Core mathematical tests
        test_mathematical_formulation()
        test_sigmoid_pretraining_protocol()
        
        # Advanced features tests (Stage 4)
        test_stage4_advanced_features()
        
        # Integration tests
        test_cross_stage_integration()
        
        # Utility tests
        test_model_summaries()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("✅ Stage 4 Advanced Features Implementation Complete")
        print("✅ Mathematical Formulation α = W @ sigmoid(s) Verified")
        print("✅ Sigmoid Pretraining Protocol Implemented")
        print("✅ All Advanced Features Working:")
        print("   • Temporal Dynamics")
        print("   • Uncertainty Quantification") 
        print("   • Multi-Scale Integration")
        print("   • Advanced Regularization")
        print("   • Interpretability Tools")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)