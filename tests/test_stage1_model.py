"""Tests for Stage 1 regulatory model components."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Import the components we're testing
from tangelo_velocity.config import TangeloConfig, get_stage_config
from tangelo_velocity.models.stage1 import Stage1RegulatoryModel
from tangelo_velocity.models.regulatory import (
    SigmoidFeatureModule, 
    LinearInteractionNetwork,
    RegulatoryNetwork
)
from tangelo_velocity.models.ode_dynamics import (
    ODEParameterPredictor,
    VelocityODE,
    ODESolver
)
from tangelo_velocity.models.loss_functions import (
    ReconstructionLoss,
    Stage1TotalLoss
)


class TestSigmoidFeatureModule:
    """Test the SigmoidFeatureModule component."""
    
    @pytest.fixture
    def sigmoid_module(self):
        """Create a test sigmoid module."""
        return SigmoidFeatureModule(
            n_genes=10,
            n_components=5,
            init_a=1.0,
            init_b=0.0
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample gene expression data."""
        torch.manual_seed(42)
        return torch.randn(20, 10)  # 20 cells, 10 genes
    
    def test_initialization(self, sigmoid_module):
        """Test proper initialization of sigmoid module."""
        assert sigmoid_module.n_genes == 10
        assert sigmoid_module.n_components == 5
        assert sigmoid_module.slopes.shape == (10, 5)
        assert sigmoid_module.biases.shape == (10, 5)
        assert sigmoid_module.weights.shape == (10, 5)
    
    def test_forward_pass(self, sigmoid_module, sample_data):
        """Test forward pass produces correct output shape."""
        output = sigmoid_module(sample_data)
        
        assert output.shape == sample_data.shape
        assert torch.all(output >= 0)  # Sigmoid outputs should be positive
        assert torch.all(output <= 1)  # Sigmoid outputs should be <= 1
    
    def test_parameter_access(self, sigmoid_module):
        """Test parameter getting and setting."""
        # Get parameters
        params = sigmoid_module.get_parameters()
        assert 'slopes' in params
        assert 'biases' in params
        assert 'weights' in params
        
        # Modify and set parameters
        new_params = {
            'slopes': torch.ones_like(params['slopes']) * 2.0,
            'biases': torch.zeros_like(params['biases']),
            'weights': torch.ones_like(params['weights']) / params['weights'].shape[-1]
        }
        
        sigmoid_module.set_parameters(new_params)
        
        # Verify parameters were updated
        updated_params = sigmoid_module.get_parameters()
        assert torch.allclose(updated_params['slopes'], new_params['slopes'])
    
    def test_cdf_pretraining(self, sigmoid_module, sample_data):
        """Test CDF-based pretraining."""
        # Mock the CDF computation to avoid complex dependencies
        with patch.object(sigmoid_module, '_compute_cdf') as mock_cdf:
            # Setup mock to return reasonable values
            x_cdf = torch.linspace(-2, 2, 20).unsqueeze(1).expand(-1, 10)
            y_cdf = torch.sigmoid(x_cdf)  # Reasonable CDF approximation
            mock_cdf.return_value = (x_cdf, y_cdf)
            
            # Test pretraining runs without error
            sigmoid_module.pretrain_on_cdf(sample_data, n_epochs=5, learning_rate=0.1)
            
            # Verify CDF computation was called
            mock_cdf.assert_called_once_with(sample_data)


class TestLinearInteractionNetwork:
    """Test the LinearInteractionNetwork component."""
    
    @pytest.fixture
    def interaction_network(self):
        """Create a test interaction network."""
        return LinearInteractionNetwork(
            n_genes=5,
            use_bias=True,
            interaction_strength=1.0
        )
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features."""
        torch.manual_seed(42)
        return torch.randn(10, 5)  # 10 cells, 5 genes
    
    @pytest.fixture
    def atac_mask(self):
        """Create a sample ATAC mask."""
        # Create a sparse regulatory network mask
        mask = torch.zeros(5, 5)
        mask[0, 1] = 1  # Gene 0 regulates gene 1
        mask[1, 2] = 1  # Gene 1 regulates gene 2
        mask[2, 0] = 1  # Gene 2 regulates gene 0 (feedback)
        mask[3, 4] = 1  # Gene 3 regulates gene 4
        return mask
    
    def test_initialization(self, interaction_network):
        """Test proper initialization."""
        assert interaction_network.n_genes == 5
        assert interaction_network.use_bias is True
        assert interaction_network.interaction_matrix.shape == (5, 5)
        assert interaction_network.bias is not None
        assert interaction_network.bias.shape == (5,)
    
    def test_atac_mask_setting(self, interaction_network, atac_mask):
        """Test ATAC mask setting and validation."""
        # Test successful mask setting
        interaction_network.set_atac_mask(atac_mask)
        assert torch.allclose(interaction_network.atac_mask, atac_mask)
        
        # Test invalid mask shape
        with pytest.raises(ValueError):
            interaction_network.set_atac_mask(torch.ones(3, 3))
    
    def test_forward_pass(self, interaction_network, sample_features, atac_mask):
        """Test forward pass with ATAC masking."""
        interaction_network.set_atac_mask(atac_mask)
        
        output = interaction_network(sample_features)
        
        assert output.shape == sample_features.shape
        
        # Test that masking is applied correctly
        masked_matrix = interaction_network.get_interaction_matrix()
        assert torch.sum((masked_matrix != 0) & (atac_mask == 0)) == 0  # No interactions where mask is 0
    
    def test_sparsity_loss(self, interaction_network, atac_mask):
        """Test sparsity loss computation."""
        interaction_network.set_atac_mask(atac_mask)
        
        sparsity_loss = interaction_network.get_sparsity_loss()
        
        assert isinstance(sparsity_loss, torch.Tensor)
        assert sparsity_loss.numel() == 1  # Scalar loss
        assert sparsity_loss >= 0  # L1 norm is non-negative


class TestODEParameterPredictor:
    """Test the ODE parameter predictor."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return get_stage_config(1)
    
    @pytest.fixture
    def parameter_predictor(self, config):
        """Create test parameter predictor."""
        return ODEParameterPredictor(
            input_dim=32,
            n_genes=8,
            config=config
        )
    
    @pytest.fixture
    def sample_features(self):
        """Create sample input features."""
        torch.manual_seed(42)
        return torch.randn(15, 32)  # 15 cells, 32 features
    
    def test_initialization(self, parameter_predictor, config):
        """Test proper initialization."""
        assert parameter_predictor.input_dim == 32
        assert parameter_predictor.n_genes == 8
        assert parameter_predictor.config == config
    
    def test_parameter_prediction(self, parameter_predictor, sample_features):
        """Test parameter prediction."""
        params = parameter_predictor(sample_features)
        
        # Check output structure
        assert 'beta' in params
        assert 'gamma' in params
        assert 'time' in params
        
        # Check shapes
        assert params['beta'].shape == (15, 8)  # batch_size x n_genes
        assert params['gamma'].shape == (15, 8)  # batch_size x n_genes
        assert params['time'].shape == (15, 1)   # batch_size x 1
        
        # Check parameter ranges
        assert torch.all(params['beta'] > 0)  # Positive splicing rates
        assert torch.all(params['gamma'] > 0)  # Positive degradation rates
        assert torch.all(params['time'] > 0)   # Positive time values


class TestVelocityODE:
    """Test the VelocityODE system."""
    
    @pytest.fixture
    def regulatory_network(self):
        """Create mock regulatory network."""
        mock_network = Mock()
        mock_network.return_value = torch.ones(5, 3) * 0.5  # Mock transcription rates
        return mock_network
    
    @pytest.fixture
    def velocity_ode(self, regulatory_network):
        """Create test ODE system."""
        return VelocityODE(
            n_genes=3,
            regulatory_network=regulatory_network
        )
    
    @pytest.fixture
    def ode_parameters(self):
        """Create test ODE parameters."""
        return {
            'beta': torch.ones(5, 3) * 0.8,    # 5 cells, 3 genes
            'gamma': torch.ones(5, 3) * 0.3
        }
    
    def test_parameter_setting(self, velocity_ode, ode_parameters):
        """Test ODE parameter setting."""
        velocity_ode.set_parameters(
            beta=ode_parameters['beta'],
            gamma=ode_parameters['gamma']
        )
        
        assert torch.allclose(velocity_ode.beta, ode_parameters['beta'])
        assert torch.allclose(velocity_ode.gamma, ode_parameters['gamma'])
    
    def test_forward_pass(self, velocity_ode, ode_parameters):
        """Test ODE forward pass."""
        velocity_ode.set_parameters(
            beta=ode_parameters['beta'],
            gamma=ode_parameters['gamma']
        )
        
        # State vector: [u, s] concatenated
        y = torch.randn(5, 6)  # 5 cells, 6 features (3 genes × 2)
        t = 0.0
        
        dy_dt = velocity_ode(t, y)
        
        assert dy_dt.shape == y.shape
        
        # Verify regulatory network was called with spliced part
        velocity_ode.regulatory_network.assert_called_once()
        call_args = velocity_ode.regulatory_network.call_args[0][0]
        expected_spliced = y[:, 3:]  # Second half is spliced
        assert torch.allclose(call_args, expected_spliced)


class TestReconstructionLoss:
    """Test the reconstruction loss function."""
    
    @pytest.fixture
    def reconstruction_loss(self):
        """Create test reconstruction loss."""
        return ReconstructionLoss(distribution="nb", theta_init=10.0)
    
    @pytest.fixture
    def predictions_and_targets(self):
        """Create test predictions and targets."""
        torch.manual_seed(42)
        
        # Generate realistic RNA count data
        pred_u = torch.abs(torch.randn(20, 10)) * 10 + 1  # Positive counts
        pred_s = torch.abs(torch.randn(20, 10)) * 20 + 2
        obs_u = torch.poisson(pred_u * 0.8 + 1)  # Add some noise
        obs_s = torch.poisson(pred_s * 0.9 + 1)
        
        return pred_u, pred_s, obs_u, obs_s
    
    def test_negative_binomial_loss(self, reconstruction_loss, predictions_and_targets):
        """Test Negative Binomial reconstruction loss."""
        pred_u, pred_s, obs_u, obs_s = predictions_and_targets
        
        loss_dict = reconstruction_loss(pred_u, pred_s, obs_u, obs_s)
        
        # Check output structure
        assert 'total' in loss_dict
        assert 'unspliced' in loss_dict
        assert 'spliced' in loss_dict
        assert 'theta' in loss_dict
        
        # Check loss values are reasonable
        assert loss_dict['total'] > 0
        assert loss_dict['unspliced'] > 0
        assert loss_dict['spliced'] > 0
        assert loss_dict['theta'] > 0
    
    def test_poisson_loss(self, predictions_and_targets):
        """Test Poisson reconstruction loss."""
        loss_fn = ReconstructionLoss(distribution="poisson")
        pred_u, pred_s, obs_u, obs_s = predictions_and_targets
        
        loss_dict = loss_fn(pred_u, pred_s, obs_u, obs_s)
        
        # Check output structure (no theta for Poisson)
        assert 'total' in loss_dict
        assert 'unspliced' in loss_dict
        assert 'spliced' in loss_dict
        assert 'theta' not in loss_dict
    
    def test_normal_loss(self, predictions_and_targets):
        """Test Normal (MSE) reconstruction loss."""
        loss_fn = ReconstructionLoss(distribution="normal")
        pred_u, pred_s, obs_u, obs_s = predictions_and_targets
        
        loss_dict = loss_fn(pred_u, pred_s, obs_u, obs_s)
        
        # Check output structure
        assert 'total' in loss_dict
        assert 'unspliced' in loss_dict
        assert 'spliced' in loss_dict


class TestStage1RegulatoryModel:
    """Test the complete Stage 1 regulatory model."""
    
    @pytest.fixture
    def config(self):
        """Create Stage 1 configuration."""
        return get_stage_config(1)
    
    @pytest.fixture
    def stage1_model(self, config):
        """Create Stage 1 model."""
        return Stage1RegulatoryModel(
            config=config,
            gene_dim=6,
            atac_dim=20
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        torch.manual_seed(42)
        
        # Generate realistic RNA count data
        n_cells, n_genes = 25, 6
        
        spliced = torch.abs(torch.randn(n_cells, n_genes)) * 20 + 5
        unspliced = torch.abs(torch.randn(n_cells, n_genes)) * 10 + 2
        
        # ATAC mask (sparse regulatory network)
        atac_mask = torch.zeros(n_genes, n_genes)
        atac_mask[0, 1] = 1
        atac_mask[1, 2] = 1
        atac_mask[2, 0] = 1
        atac_mask[3, 4] = 1
        atac_mask[4, 5] = 1
        atac_mask[5, 3] = 1
        
        return {
            'spliced': spliced,
            'unspliced': unspliced,
            'atac_mask': atac_mask
        }
    
    def test_initialization(self, stage1_model, config):
        """Test proper model initialization."""
        assert stage1_model.development_stage == 1
        assert stage1_model.gene_dim == 6
        assert stage1_model.atac_dim == 20
        assert stage1_model.config == config
        
        # Check component initialization
        assert hasattr(stage1_model, 'regulatory_network')
        assert hasattr(stage1_model, 'feature_encoder')
        assert hasattr(stage1_model, 'ode_parameter_predictor')
        assert hasattr(stage1_model, 'velocity_ode')
        assert hasattr(stage1_model, 'ode_solver')
        assert hasattr(stage1_model, 'loss_fn')
    
    def test_atac_mask_setting(self, stage1_model, sample_data):
        """Test ATAC mask setting."""
        atac_mask = sample_data['atac_mask']
        
        stage1_model.set_atac_mask(atac_mask)
        
        # Verify mask was set correctly
        assert torch.allclose(stage1_model.atac_mask, atac_mask)
    
    @patch('tangelo_velocity.models.ode_dynamics.ODESolver.solve')
    def test_forward_pass(self, mock_solve, stage1_model, sample_data):
        """Test forward pass through Stage 1 model."""
        # Mock ODE solver to avoid complex dependencies
        mock_solution = {
            'final_state': torch.cat([
                sample_data['unspliced'] * 0.9,  # Mock predicted unspliced
                sample_data['spliced'] * 1.1    # Mock predicted spliced
            ], dim=1),
            'times': torch.linspace(0, 1, 10).unsqueeze(0).expand(25, -1),
            'status': torch.zeros(25)
        }
        mock_solve.return_value = mock_solution
        
        # Set ATAC mask
        stage1_model.set_atac_mask(sample_data['atac_mask'])
        
        # Forward pass
        outputs = stage1_model(
            spliced=sample_data['spliced'],
            unspliced=sample_data['unspliced']
        )
        
        # Check output structure
        assert 'pred_unspliced' in outputs
        assert 'pred_spliced' in outputs
        assert 'velocity' in outputs
        assert 'ode_params' in outputs
        assert 'transcription_rates' in outputs
        
        # Check output shapes
        assert outputs['pred_unspliced'].shape == sample_data['unspliced'].shape
        assert outputs['pred_spliced'].shape == sample_data['spliced'].shape
        assert outputs['velocity'].shape == sample_data['spliced'].shape
    
    def test_loss_computation(self, stage1_model, sample_data):
        """Test loss computation."""
        # Mock forward pass outputs
        outputs = {
            'pred_unspliced': sample_data['unspliced'] * 0.9,
            'pred_spliced': sample_data['spliced'] * 1.1,
            'velocity': torch.randn_like(sample_data['spliced']) * 0.1,
            'ode_params': {
                'beta': torch.ones(25, 6) * 0.8,
                'gamma': torch.ones(25, 6) * 0.3
            }
        }
        
        targets = {
            'unspliced': sample_data['unspliced'],
            'spliced': sample_data['spliced']
        }
        
        loss = stage1_model.compute_loss(outputs, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Scalar loss
        assert loss > 0  # Positive loss
    
    def test_model_summary(self, stage1_model):
        """Test model summary generation."""
        summary = stage1_model.get_model_summary()
        
        assert isinstance(summary, str)
        assert "Stage 1 Regulatory Model Summary" in summary
        assert "Development Stage: 1" in summary
        assert "Genes: 6" in summary
    
    def test_pretrain_sigmoid(self, stage1_model, sample_data):
        """Test sigmoid pretraining."""
        # Mock the pretraining to avoid complex dependencies
        with patch.object(stage1_model.regulatory_network, 'pretrain_sigmoid') as mock_pretrain:
            stage1_model.pretrain_sigmoid_features(
                sample_data['spliced'],
                n_epochs=5,
                learning_rate=0.1
            )
            
            mock_pretrain.assert_called_once_with(
                sample_data['spliced'],
                n_epochs=5,
                learning_rate=0.1
            )


# Integration tests
class TestStage1Integration:
    """Integration tests for Stage 1 components working together."""
    
    @pytest.fixture
    def config(self):
        """Create Stage 1 configuration."""
        config = get_stage_config(1)
        # Reduce epochs for faster testing
        config.training.n_epochs = 3
        config.training.log_interval = 1
        return config
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic multimodal data."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        n_cells, n_genes = 50, 8
        
        # Generate correlated spliced/unspliced data
        base_expression = torch.randn(n_cells, n_genes).abs() * 10 + 2
        spliced = base_expression + torch.randn(n_cells, n_genes) * 2
        unspliced = base_expression * 0.3 + torch.randn(n_cells, n_genes) * 1
        
        # Ensure positive counts
        spliced = torch.clamp(spliced, min=0.1)
        unspliced = torch.clamp(unspliced, min=0.1)
        
        # Create sparse ATAC mask
        atac_mask = torch.zeros(n_genes, n_genes)
        # Add some regulatory connections
        for i in range(n_genes - 1):
            atac_mask[i, i + 1] = 1  # Sequential regulation
        atac_mask[n_genes - 1, 0] = 1  # Close the loop
        
        # Add some additional random connections
        for _ in range(n_genes // 2):
            i, j = np.random.choice(n_genes, 2, replace=False)
            atac_mask[i, j] = 1
        
        return {
            'spliced': spliced,
            'unspliced': unspliced,
            'atac_mask': atac_mask
        }
    
    @patch('tangelo_velocity.models.ode_dynamics.ODESolver.solve')
    def test_end_to_end_training(self, mock_solve, config, synthetic_data):
        """Test end-to-end training process."""
        # Mock ODE solver for faster testing
        def mock_solve_fn(ode_system, y0, t_span, ode_params):
            # Simple mock: return slightly modified input
            batch_size = y0.shape[0]
            n_genes = y0.shape[1] // 2
            
            return {
                'final_state': y0 + torch.randn_like(y0) * 0.1,
                'times': torch.linspace(t_span[0], t_span[1], 10).unsqueeze(0).expand(batch_size, -1),
                'status': torch.zeros(batch_size)
            }
        
        mock_solve.side_effect = mock_solve_fn
        
        # Create model
        model = Stage1RegulatoryModel(
            config=config,
            gene_dim=8,
            atac_dim=20
        )
        
        # Set ATAC mask
        model.set_atac_mask(synthetic_data['atac_mask'])
        
        # Prepare training data
        training_data = {
            'spliced': synthetic_data['spliced'],
            'unspliced': synthetic_data['unspliced']
        }
        
        # Test training (should not raise errors)
        model.fit(training_data)
        
        # Test prediction
        velocity = model.predict_velocity()
        assert velocity.shape == synthetic_data['spliced'].shape
        
        # Test parameter extraction
        ode_params = model.get_ode_parameters()
        assert 'beta' in ode_params
        assert 'gamma' in ode_params
        assert 'time' in ode_params
        
        # Test interaction network
        interaction_matrix = model.get_interaction_network()
        assert interaction_matrix.shape == (8, 8)
    
    def test_model_factory(self, config):
        """Test model creation through factory function."""
        from tangelo_velocity.models import get_velocity_model
        
        model = get_velocity_model(
            config=config,
            gene_dim=5,
            atac_dim=15
        )
        
        assert isinstance(model, Stage1RegulatoryModel)
        assert model.development_stage == 1
        assert model.gene_dim == 5
        assert model.atac_dim == 15