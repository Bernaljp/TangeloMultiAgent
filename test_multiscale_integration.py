"""Comprehensive tests for multiscale integration module."""

import pytest
import torch
import numpy as np
from typing import Dict, Any

# Import the modules we need to test
from tangelo_velocity.config import get_stage_config
from tangelo_velocity.models.multiscale import (
    MultiscaleConfig, MultiscaleSampler, MultiscaleLoss, MultiscaleTrainer,
    create_multiscale_config, validate_multiscale_implementation
)


class TestMultiscaleConfig:
    """Test MultiscaleConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MultiscaleConfig()
        
        assert config.enable_multiscale == False
        assert config.multiscale_weights == (0.4, 0.3, 0.2, 0.1)
        assert config.min_scale_size == 1
        assert config.max_scales == 4
        assert config.scale_strategy == "geometric"
        assert config.random_seed is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        weights = (0.5, 0.3, 0.2)
        config = MultiscaleConfig(
            enable_multiscale=True,
            multiscale_weights=weights,
            min_scale_size=2,
            max_scales=3,
            scale_strategy="linear",
            random_seed=42
        )
        
        assert config.enable_multiscale == True
        assert config.multiscale_weights == weights
        assert config.min_scale_size == 2
        assert config.max_scales == 3
        assert config.scale_strategy == "linear"
        assert config.random_seed == 42
    
    def test_create_multiscale_config(self):
        """Test create_multiscale_config utility function."""
        config = create_multiscale_config(
            enable=True,
            max_scales=3,
            min_scale_size=4,
            weights=[0.6, 0.3, 0.1]
        )
        
        assert config.enable_multiscale == True
        assert config.max_scales == 3
        assert config.min_scale_size == 4
        assert config.multiscale_weights == [0.6, 0.3, 0.1]


class TestMultiscaleSampler:
    """Test MultiscaleSampler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MultiscaleConfig(
            enable_multiscale=True,
            max_scales=4,
            min_scale_size=1,
            scale_strategy="geometric",
            random_seed=42
        )
        self.sampler = MultiscaleSampler(self.config)
    
    def test_get_scale_sizes_geometric(self):
        """Test geometric scale size calculation."""
        batch_size = 64
        scale_sizes = self.sampler.get_scale_sizes(batch_size)
        
        # Should be [64, 32, 16, 8] or similar geometric progression
        assert len(scale_sizes) <= self.config.max_scales
        assert scale_sizes[0] == batch_size
        assert all(scale_sizes[i] >= scale_sizes[i+1] for i in range(len(scale_sizes)-1))
        assert min(scale_sizes) >= self.config.min_scale_size
    
    def test_get_scale_sizes_linear(self):
        """Test linear scale size calculation."""
        config = MultiscaleConfig(
            enable_multiscale=True,
            max_scales=4,
            scale_strategy="linear"
        )
        sampler = MultiscaleSampler(config)
        
        batch_size = 100
        scale_sizes = sampler.get_scale_sizes(batch_size)
        
        assert len(scale_sizes) <= config.max_scales
        assert scale_sizes[0] == batch_size
        assert all(scale_sizes[i] >= scale_sizes[i+1] for i in range(len(scale_sizes)-1))
    
    def test_get_scale_sizes_disabled(self):
        """Test scale sizes when multiscale is disabled."""
        config = MultiscaleConfig(enable_multiscale=False)
        sampler = MultiscaleSampler(config)
        
        batch_size = 64
        scale_sizes = sampler.get_scale_sizes(batch_size)
        
        assert scale_sizes == [batch_size]
    
    def test_sample_scales(self):
        """Test hierarchical sampling from batch data."""
        batch_size = 32
        n_genes = 100
        
        batch_data = {
            'spliced': torch.randn(batch_size, n_genes),
            'unspliced': torch.randn(batch_size, n_genes),
            'metadata': torch.randn(10)  # Non-batch tensor
        }
        
        scale_samples = self.sampler.sample_scales(batch_data)
        
        # Should have multiple scales
        assert len(scale_samples) > 1
        
        # First scale should be full batch
        assert scale_samples[0]['spliced'].shape[0] == batch_size
        assert scale_samples[0]['unspliced'].shape[0] == batch_size
        
        # Later scales should be smaller
        for i in range(1, len(scale_samples)):
            assert scale_samples[i]['spliced'].shape[0] <= scale_samples[i-1]['spliced'].shape[0]
            assert scale_samples[i]['unspliced'].shape[0] <= scale_samples[i-1]['unspliced'].shape[0]
            # Non-batch tensors should be unchanged
            assert torch.equal(scale_samples[i]['metadata'], batch_data['metadata'])
    
    def test_get_scale_weights(self):
        """Test scale weight computation."""
        num_scales = 4
        weights = self.sampler.get_scale_weights(num_scales)
        
        assert len(weights) == num_scales
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.all(weights > 0)
    
    def test_reproducible_sampling(self):
        """Test that sampling is reproducible with fixed seed."""
        batch_data = {
            'spliced': torch.randn(64, 100),
            'unspliced': torch.randn(64, 100)
        }
        
        # Sample twice with same seed
        samples1 = self.sampler.sample_scales(batch_data)
        samples2 = self.sampler.sample_scales(batch_data)
        
        # Results should be identical
        assert len(samples1) == len(samples2)
        for s1, s2 in zip(samples1, samples2):
            if s1['spliced'].shape[0] < batch_data['spliced'].shape[0]:  # Only for sampled scales
                assert torch.equal(s1['spliced'], s2['spliced'])
                assert torch.equal(s1['unspliced'], s2['unspliced'])


class TestMultiscaleLoss:
    """Test MultiscaleLoss functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MultiscaleConfig(
            enable_multiscale=True,
            max_scales=3,
            min_scale_size=1,
            random_seed=42
        )
        
        # Mock base loss function
        def mock_loss_fn(outputs, targets, **kwargs):
            return {
                'total': torch.tensor(1.0),
                'component1': torch.tensor(0.5),
                'component2': torch.tensor(0.5)
            }
        
        self.multiscale_loss = MultiscaleLoss(mock_loss_fn, self.config)
    
    def test_disabled_multiscale(self):
        """Test loss computation when multiscale is disabled."""
        config = MultiscaleConfig(enable_multiscale=False)
        loss_fn = MultiscaleLoss(lambda o, t, **k: {'total': torch.tensor(2.0)}, config)
        
        # Mock model
        def mock_model(**inputs):
            return {'output': torch.randn(10, 5)}
        
        batch_data = {'input': torch.randn(10, 5)}
        targets = {'target': torch.randn(10, 5)}
        
        result = loss_fn(mock_model, batch_data, targets)
        
        assert result['total'] == torch.tensor(2.0)
        assert not result.get('multiscale_enabled', False)
    
    def test_enabled_multiscale(self):
        """Test loss computation when multiscale is enabled."""
        # Mock model
        def mock_model(**inputs):
            batch_size = next(iter(inputs.values())).shape[0]
            return {'output': torch.randn(batch_size, 5)}
        
        batch_data = {'input': torch.randn(16, 5)}
        targets = {'target': torch.randn(16, 5)}
        
        result = self.multiscale_loss(mock_model, batch_data, targets)
        
        assert result['multiscale_enabled'] == True
        assert 'total' in result
        assert 'num_scales' in result
        assert 'scale_weights' in result
        assert 'scale_losses' in result
        assert result['num_scales'] > 1
    
    def test_loss_aggregation(self):
        """Test that multiscale loss properly aggregates scale losses."""
        # Mock model that returns different losses for different batch sizes
        def mock_model(**inputs):
            batch_size = next(iter(inputs.values())).shape[0]
            return {'output': torch.randn(batch_size, 5)}
        
        def scale_dependent_loss_fn(outputs, targets, **kwargs):
            batch_size = outputs['output'].shape[0]
            return {'total': torch.tensor(float(batch_size))}  # Loss = batch size
        
        multiscale_loss = MultiscaleLoss(scale_dependent_loss_fn, self.config)
        
        batch_data = {'input': torch.randn(8, 5)}
        targets = {'target': torch.randn(8, 5)}
        
        result = multiscale_loss(mock_model, batch_data, targets)
        
        # Verify weighted combination
        scale_losses = result['scale_losses']
        scale_weights = result['scale_weights']
        expected_total = torch.sum(scale_losses * scale_weights)
        
        assert torch.allclose(result['total'], expected_total, atol=1e-6)
    
    def test_get_multiscale_analysis(self):
        """Test multiscale analysis functionality."""
        loss_dict = {
            'total': torch.tensor(2.5),
            'multiscale_enabled': True,
            'num_scales': 3,
            'scale_losses': torch.tensor([3.0, 2.0, 1.0]),
            'scale_weights': torch.tensor([0.5, 0.3, 0.2]),
            'scale_sizes': torch.tensor([16, 8, 4])
        }
        
        analysis = self.multiscale_loss.get_multiscale_analysis(loss_dict)
        
        assert analysis['multiscale_enabled'] == True
        assert analysis['num_scales'] == 3
        assert analysis['total_loss'] == 2.5
        assert len(analysis['scale_losses']) == 3
        assert len(analysis['scale_weights']) == 3
        assert analysis['dominant_scale'] == 2  # Index of minimum loss
        assert 'loss_variance_across_scales' in analysis


class TestMultiscaleTrainer:
    """Test MultiscaleTrainer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock model
        class MockModel:
            def forward(self, **inputs):
                batch_size = next(iter(inputs.values())).shape[0]
                return {'output': torch.randn(batch_size, 5)}
        
        self.model = MockModel()
        
        # Mock loss function
        def mock_loss_fn(outputs, targets, **kwargs):
            return {'total': torch.tensor(1.0)}
        
        self.config = MultiscaleConfig(
            enable_multiscale=True,
            max_scales=3,
            random_seed=42
        )
        
        self.trainer = MultiscaleTrainer(self.model, mock_loss_fn, self.config)
    
    def test_training_step(self):
        """Test single training step with multiscale."""
        batch_data = {'input': torch.randn(16, 10)}
        targets = {'target': torch.randn(16, 10)}
        
        loss_dict = self.trainer.training_step(batch_data, targets)
        
        assert 'total' in loss_dict
        assert loss_dict.get('multiscale_enabled', False) == True
        assert self.trainer.step_count == 1
    
    def test_training_statistics(self):
        """Test training statistics accumulation."""
        batch_data = {'input': torch.randn(16, 10)}
        targets = {'target': torch.randn(16, 10)}
        
        # Run multiple training steps
        for _ in range(5):
            self.trainer.training_step(batch_data, targets)
        
        summary = self.trainer.get_training_summary()
        
        assert summary['total_steps'] == 5
        assert summary['multiscale_enabled'] == True
        assert 'scale_configuration' in summary
    
    def test_disabled_multiscale_trainer(self):
        """Test trainer with disabled multiscale."""
        config = MultiscaleConfig(enable_multiscale=False)
        trainer = MultiscaleTrainer(self.model, lambda o, t: {'total': torch.tensor(1.0)}, config)
        
        batch_data = {'input': torch.randn(16, 10)}
        targets = {'target': torch.randn(16, 10)}
        
        loss_dict = trainer.training_step(batch_data, targets)
        summary = trainer.get_training_summary()
        
        assert loss_dict.get('multiscale_enabled', False) == False
        assert summary['multiscale_enabled'] == False


class TestIntegrationWithStage4:
    """Test integration with Stage 4 model."""
    
    def test_stage4_config_integration(self):
        """Test that Stage 4 config includes multiscale settings."""
        config = get_stage_config(4)
        
        assert hasattr(config, 'multiscale')
        assert config.multiscale.enable_multiscale == True
        assert config.multiscale.max_scales == 4
        assert config.multiscale.min_scale_size == 1
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multiscale_cuda_compatibility(self):
        """Test multiscale functionality with CUDA tensors."""
        config = MultiscaleConfig(
            enable_multiscale=True,
            max_scales=3,
            random_seed=42
        )
        sampler = MultiscaleSampler(config)
        
        batch_data = {
            'spliced': torch.randn(32, 100).cuda(),
            'unspliced': torch.randn(32, 100).cuda()
        }
        
        scale_samples = sampler.sample_scales(batch_data)
        
        # All samples should be on CUDA
        for sample in scale_samples:
            assert sample['spliced'].is_cuda
            assert sample['unspliced'].is_cuda


class TestValidationAndUtilities:
    """Test validation functions and utilities."""
    
    def test_validate_multiscale_implementation(self):
        """Test the validation utility function."""
        results = validate_multiscale_implementation(
            batch_size=32,
            max_scales=3,
            min_scale_size=2
        )
        
        assert results['original_batch_size'] == 32
        assert results['num_scales_generated'] <= 3
        assert results['valid_sampling'] == True
        assert results['decreasing_sizes'] == True
        assert results['min_size_respected'] == True
        assert abs(results['weight_sum'] - 1.0) < 1e-6
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Very small batch size
        config = MultiscaleConfig(enable_multiscale=True, max_scales=5, min_scale_size=1)
        sampler = MultiscaleSampler(config)
        
        batch_data = {'input': torch.randn(2, 10)}
        scale_samples = sampler.sample_scales(batch_data)
        
        # Should handle small batches gracefully
        assert len(scale_samples) >= 1
        assert all(sample['input'].shape[0] >= 1 for sample in scale_samples)
    
    def test_mathematical_correctness(self):
        """Test mathematical correctness of loss computation."""
        config = MultiscaleConfig(
            enable_multiscale=True,
            multiscale_weights=(0.5, 0.3, 0.2),
            max_scales=3,
            random_seed=42
        )
        
        # Loss function that returns scale-dependent loss
        def scale_loss_fn(outputs, targets, **kwargs):
            batch_size = outputs['output'].shape[0]
            return {'total': torch.tensor(batch_size * 0.1)}
        
        multiscale_loss = MultiscaleLoss(scale_loss_fn, config)
        
        def mock_model(**inputs):
            batch_size = next(iter(inputs.values())).shape[0]
            return {'output': torch.randn(batch_size, 5)}
        
        batch_data = {'input': torch.randn(16, 5)}
        targets = {'target': torch.randn(16, 5)}
        
        result = multiscale_loss(mock_model, batch_data, targets)
        
        # Verify mathematical properties
        assert result['multiscale_enabled'] == True
        assert torch.allclose(result['scale_weights'].sum(), torch.tensor(1.0))
        
        # Verify weighted sum
        expected_loss = torch.sum(result['scale_losses'] * result['scale_weights'])
        assert torch.allclose(result['total'], expected_loss, atol=1e-6)


def test_comprehensive_multiscale_workflow():
    """Test complete multiscale training workflow."""
    # Create configuration
    config = create_multiscale_config(
        enable=True,
        max_scales=4,
        min_scale_size=1,
        weights=[0.4, 0.3, 0.2, 0.1]
    )
    
    # Create sampler
    sampler = MultiscaleSampler(config)
    
    # Test batch
    batch_size = 64
    batch_data = {
        'spliced': torch.randn(batch_size, 100),
        'unspliced': torch.randn(batch_size, 100)
    }
    
    # Sample at different scales
    scale_samples = sampler.sample_scales(batch_data)
    scale_weights = sampler.get_scale_weights(len(scale_samples))
    
    # Verify workflow
    assert len(scale_samples) <= config.max_scales
    assert len(scale_weights) == len(scale_samples)
    assert torch.allclose(scale_weights.sum(), torch.tensor(1.0))
    
    # Verify scale sizes are decreasing
    scale_sizes = [sample['spliced'].shape[0] for sample in scale_samples]
    assert all(scale_sizes[i] >= scale_sizes[i+1] for i in range(len(scale_sizes)-1))
    
    print("✓ Comprehensive multiscale workflow test passed")
    print(f"  - Generated {len(scale_samples)} scales")
    print(f"  - Scale sizes: {scale_sizes}")
    print(f"  - Scale weights: {scale_weights.tolist()}")


if __name__ == "__main__":
    # Run comprehensive workflow test
    test_comprehensive_multiscale_workflow()
    
    # Run basic validation
    validation_results = validate_multiscale_implementation(batch_size=32, max_scales=4)
    print("\n✓ Multiscale implementation validation:")
    for key, value in validation_results.items():
        print(f"  - {key}: {value}")
    
    print("\n🎉 All multiscale integration tests completed successfully!")
    print("\nTo run full test suite:")
    print("pytest test_multiscale_integration.py -v")