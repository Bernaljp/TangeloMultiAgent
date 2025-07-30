#!/usr/bin/env python3
"""
Test script for Stage 3 Integrated Model implementation.

This script validates the complete Stage 3 implementation including:
- Model instantiation and component initialization
- Multi-modal forward pass functionality
- Integration of regulatory and graph components
- Loss computation and optimization
- ODE system integration
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: torch_geometric not available. Some tests will be skipped.")

# Import the Stage 3 model and configuration
from tangelo_velocity.models import get_velocity_model, Stage3IntegratedModel
from tangelo_velocity.config import get_stage_config


def create_synthetic_data(n_cells: int = 100, n_genes: int = 50, n_atac: int = 20) -> Dict[str, torch.Tensor]:
    """Create synthetic data for testing Stage 3 model."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic RNA data
    spliced = torch.abs(torch.randn(n_cells, n_genes)) * 10 + 1
    unspliced = torch.abs(torch.randn(n_cells, n_genes)) * 5 + 0.5
    
    # Create synthetic spatial coordinates
    spatial_coords = torch.randn(n_cells, 2) * 10
    
    # Create synthetic ATAC mask (gene-gene interactions)
    atac_mask = torch.rand(n_genes, n_genes) > 0.7  # Sparse connectivity
    atac_mask = atac_mask.float()
    
    # Make symmetric and add diagonal
    atac_mask = (atac_mask + atac_mask.T) / 2
    atac_mask.fill_diagonal_(1.0)
    
    return {
        'spliced': spliced,
        'unspliced': unspliced,
        'spatial_coords': spatial_coords,
        'atac_mask': atac_mask,
        'n_genes': n_genes,
        'n_atac': n_atac
    }


def create_synthetic_graphs(n_cells: int, device: torch.device) -> Dict[str, Data]:
    """Create synthetic spatial and expression graphs."""
    
    if not HAS_TORCH_GEOMETRIC:
        return {}
    
    # Create spatial k-NN graph (each cell connected to 5 nearest neighbors)
    k_spatial = min(5, n_cells - 1)
    spatial_edges = []
    
    for i in range(n_cells):
        # Connect to k random neighbors for simplicity
        neighbors = np.random.choice(
            [j for j in range(n_cells) if j != i], 
            size=min(k_spatial, n_cells - 1), 
            replace=False
        )
        for neighbor in neighbors:
            spatial_edges.append([i, neighbor])
    
    spatial_edge_index = torch.tensor(spatial_edges, device=device).T
    spatial_graph = Data(edge_index=spatial_edge_index, num_nodes=n_cells)
    
    # Create expression similarity graph
    k_expr = min(8, n_cells - 1)
    expr_edges = []
    
    for i in range(n_cells):
        neighbors = np.random.choice(
            [j for j in range(n_cells) if j != i],
            size=min(k_expr, n_cells - 1),
            replace=False
        )
        for neighbor in neighbors:
            expr_edges.append([i, neighbor])
    
    expr_edge_index = torch.tensor(expr_edges, device=device).T
    expr_graph = Data(edge_index=expr_edge_index, num_nodes=n_cells)
    
    return {
        'spatial_graph': spatial_graph,
        'expression_graph': expr_graph
    }


def test_model_initialization():
    """Test Stage 3 model initialization."""
    print("Testing Stage 3 model initialization...")
    
    # Create configuration
    config = get_stage_config(3)
    
    # Test data dimensions
    n_genes, n_atac = 50, 20
    
    try:
        # Test direct model creation
        model = Stage3IntegratedModel(config, n_genes, n_atac)
        print("✓ Direct model creation successful")
        
        # Test factory function
        factory_model = get_velocity_model(config, n_genes, n_atac)
        print("✓ Factory model creation successful")
        
        # Check model components
        assert hasattr(model, 'regulatory_network'), "Missing regulatory network"
        assert hasattr(model, 'spatial_encoder'), "Missing spatial encoder"
        assert hasattr(model, 'expression_encoder'), "Missing expression encoder"
        assert hasattr(model, 'attention_fusion'), "Missing attention fusion"
        assert hasattr(model, 'integrated_ode'), "Missing integrated ODE"
        print("✓ All required components present")
        
        return model
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        raise


def test_forward_pass(model, data):
    """Test Stage 3 forward pass with multi-modal inputs."""
    print("\nTesting Stage 3 forward pass...")
    
    device = torch.device('cpu')
    model = model.to(device)
    
    # Prepare inputs
    spliced = data['spliced'].to(device)
    unspliced = data['unspliced'].to(device)
    spatial_coords = data['spatial_coords'].to(device)
    atac_mask = data['atac_mask'].to(device)
    
    # Create graphs if available
    graphs = create_synthetic_graphs(spliced.shape[0], device)
    
    try:
        # Test forward pass with all inputs
        if HAS_TORCH_GEOMETRIC and graphs:
            outputs = model(
                spliced=spliced,
                unspliced=unspliced,
                spatial_graph=graphs['spatial_graph'],
                expression_graph=graphs['expression_graph'],
                spatial_coords=spatial_coords,
                atac_mask=atac_mask
            )
        else:
            # Test without graphs (will create identity graphs internally)
            outputs = model(
                spliced=spliced,
                unspliced=unspliced,
                spatial_coords=spatial_coords,
                atac_mask=atac_mask
            )
        
        print("✓ Forward pass successful")
        
        # Check required outputs
        required_outputs = [
            'velocity', 'pred_unspliced', 'pred_spliced', 
            'transcription_rates', 'integrated_features',
            'attention_weights', 'ode_params'
        ]
        
        for output in required_outputs:
            assert output in outputs, f"Missing output: {output}"
        
        print("✓ All required outputs present")
        
        # Check output shapes
        batch_size, n_genes = spliced.shape
        assert outputs['velocity'].shape == (batch_size, n_genes), "Incorrect velocity shape"
        assert outputs['transcription_rates'].shape == (batch_size, n_genes), "Incorrect transcription rates shape"
        assert outputs['attention_weights'].shape[0] == batch_size, "Incorrect attention weights shape"
        
        print("✓ Output shapes correct")
        
        return outputs
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise


def test_integration_components(model, outputs):
    """Test the integration-specific components."""
    print("\nTesting integration components...")
    
    try:
        # Test attention weights (should sum to 1)
        attention_weights = outputs['attention_weights']
        attention_sums = torch.sum(attention_weights, dim=1)
        assert torch.allclose(attention_sums, torch.ones_like(attention_sums), atol=1e-5), \
            "Attention weights don't sum to 1"
        print("✓ Attention weights properly normalized")
        
        # Test integration weights
        integration_weights = outputs['integration_weights']
        assert len(integration_weights) == 2, "Should have 2 integration weights"
        assert torch.allclose(torch.sum(integration_weights), torch.tensor(1.0), atol=1e-5), \
            "Integration weights don't sum to 1"
        print("✓ Integration weights properly normalized")
        
        # Test transcription rate integration
        regulatory_rates = outputs['regulatory_rates']
        graph_rates = outputs['graph_transcription_rates']
        integrated_rates = outputs['transcription_rates']
        
        # Check that integrated rates are positive
        assert torch.all(integrated_rates >= 0), "Transcription rates should be positive"
        print("✓ Integrated transcription rates are positive")
        
        # Test integration analysis
        analysis = model.get_integration_analysis()
        assert 'attention' in analysis, "Missing attention analysis"
        assert 'integration_weights' in analysis, "Missing integration weights analysis"
        print("✓ Integration analysis successful")
        
        return analysis
        
    except Exception as e:
        print(f"✗ Integration component test failed: {e}")
        raise


def test_loss_computation(model, outputs, data):
    """Test comprehensive loss computation."""
    print("\nTesting loss computation...")
    
    device = outputs['velocity'].device
    
    # Prepare targets
    targets = {
        'spliced': data['spliced'].to(device),
        'unspliced': data['unspliced'].to(device)
    }
    
    try:
        # Test loss computation
        total_loss = model.compute_loss(outputs, targets)
        
        print("✓ Loss computation successful")
        
        # Check loss value
        assert torch.isfinite(total_loss), "Loss is not finite"
        assert total_loss >= 0, "Loss should be non-negative"
        print("✓ Loss value is valid")
        
        # Test detailed loss breakdown
        loss_dict = model.get_loss_dict()
        
        expected_components = [
            'regulatory_total', 'graph_total', 'attention_loss',
            'integration_loss', 'consistency_loss', 'total'
        ]
        
        for component in expected_components:
            assert component in loss_dict, f"Missing loss component: {component}"
        
        print("✓ Loss breakdown complete")
        
        return total_loss, loss_dict
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        raise


def test_ode_integration(model, outputs):
    """Test ODE system integration."""  
    print("\nTesting ODE integration...")
    
    try:
        # Test ODE parameter access
        ode_params = model._get_ode_parameters()
        
        required_params = ['beta', 'gamma']
        for param in required_params:
            assert param in ode_params, f"Missing ODE parameter: {param}"
            assert torch.all(ode_params[param] > 0), f"ODE parameter {param} should be positive"
        
        print("✓ ODE parameters valid")
        
        # Test interaction network access
        interaction_matrix = model._get_interaction_network()
        n_genes = model.gene_dim
        assert interaction_matrix.shape == (n_genes, n_genes), "Incorrect interaction matrix shape"
        print("✓ Interaction network accessible")
        
        # Test transcription rate access
        spliced = outputs['pred_spliced']
        transcription_rates = model._get_transcription_rates(spliced)
        assert transcription_rates.shape == spliced.shape, "Incorrect transcription rates shape"
        print("✓ Transcription rates accessible")
        
    except Exception as e:
        print(f"✗ ODE integration test failed: {e}")
        raise


def test_mathematical_correctness(outputs):
    """Test mathematical correctness of the α = W(sigmoid(s)) + G(graph) formulation."""
    print("\nTesting mathematical correctness...")
    
    try:
        # Extract components
        regulatory_rates = outputs['regulatory_rates']
        graph_rates = outputs['graph_transcription_rates'] 
        integrated_rates = outputs['transcription_rates']
        integration_weights = outputs['integration_weights']
        
        # Manual integration computation
        expected_integrated = (
            integration_weights[0] * regulatory_rates + 
            integration_weights[1] * graph_rates
        )
        
        # Apply softplus as in the model
        expected_integrated = F.softplus(expected_integrated)
        
        # Check if integrated rates match expected computation (approximately)
        relative_error = torch.abs(integrated_rates - expected_integrated) / (expected_integrated + 1e-8)
        max_error = torch.max(relative_error).item()
        
        assert max_error < 0.1, f"Integration formula error too large: {max_error}"
        print(f"✓ Mathematical formulation correct (max error: {max_error:.6f})")
        
        # Check that integration preserves important properties
        assert torch.all(integrated_rates > 0), "Integrated rates should be positive"
        print("✓ Integration preserves positivity")
        
    except Exception as e:
        print(f"✗ Mathematical correctness test failed: {e}")
        raise


def test_model_summary(model):
    """Test model summary and analysis functions."""
    print("\nTesting model summary and analysis...")
    
    try:
        # Test model summary
        summary = model.get_model_summary()
        assert "Stage 3 Integrated Model Summary" in summary, "Missing model summary header"
        assert "Parameters:" in summary, "Missing parameter information"
        assert "Integration Features:" in summary, "Missing integration features"
        print("✓ Model summary generated successfully")
        
        # Print key information
        lines = summary.split('\n')
        for line in lines:
            if 'Total:' in line and 'Parameters:' not in line:
                print(f"  {line.strip()}")
        
    except Exception as e:
        print(f"✗ Model summary test failed: {e}")
        raise


def run_comprehensive_test():
    """Run comprehensive Stage 3 integration test."""
    print("=" * 60)
    print("STAGE 3 INTEGRATED MODEL COMPREHENSIVE TEST")
    print("=" * 60)
    
    try:
        # 1. Create synthetic data
        print("\nCreating synthetic test data...")
        data = create_synthetic_data(n_cells=50, n_genes=30, n_atac=15)
        print("✓ Synthetic data created")
        
        # 2. Test model initialization
        model = test_model_initialization()
        
        # 3. Test forward pass
        outputs = test_forward_pass(model, data)
        
        # 4. Test integration components
        analysis = test_integration_components(model, outputs)
        
        # 5. Test loss computation
        total_loss, loss_dict = test_loss_computation(model, outputs, data)
        
        # 6. Test ODE integration
        test_ode_integration(model, outputs)
        
        # 7. Test mathematical correctness
        test_mathematical_correctness(outputs)
        
        # 8. Test model summary
        test_model_summary(model)
        
        # Summary of results
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print("✓ All Stage 3 integration tests PASSED")
        print(f"✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"✓ Total loss: {total_loss.item():.6f}")
        print(f"✓ Attention balance: Reg={analysis['attention']['regulatory_weight_mean']:.3f}, Graph={analysis['attention']['graph_weight_mean']:.3f}")
        print(f"✓ Integration weights: Reg={analysis['integration_weights']['regulatory']:.3f}, Graph={analysis['integration_weights']['graph']:.3f}")
        
        if 'component_correlation' in analysis:
            print(f"✓ Component correlation: {analysis['component_correlation']:.3f}")
        
        print("\nStage 3 implementation is COMPLETE and FUNCTIONAL! 🎉")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        print("\nStage 3 implementation needs debugging.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)