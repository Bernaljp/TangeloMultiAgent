"""ODE dynamics and parameter prediction for Stage 1 models."""

from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchode as to

from .base import MLP, safe_log


class ODEParameterPredictor(nn.Module):
    """
    Predicts cell-specific ODE parameters (beta, gamma, time) from latent representations.
    
    This module uses MLPs to predict the kinetic parameters needed for the
    RNA velocity ODE system from learned cell representations.
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of input features (e.g., from graph encoders).
    n_genes : int
        Number of genes.
    config : TangeloConfig
        Configuration object containing ODE parameters.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_genes: int,
        config
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_genes = n_genes
        self.config = config
        
        # Parameter prediction networks
        self.beta_predictor = MLP(
            input_dim=input_dim,
            hidden_dims=(128, 64),
            output_dim=n_genes,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
        
        self.gamma_predictor = MLP(
            input_dim=input_dim,
            hidden_dims=(128, 64),
            output_dim=n_genes,
            activation="relu", 
            dropout=0.1,
            batch_norm=True
        )
        
        self.time_predictor = MLP(
            input_dim=input_dim,
            hidden_dims=(128, 64),
            output_dim=1,
            activation="relu",
            dropout=0.1,
            batch_norm=True
        )
        
        # Initialize parameters within biological ranges
        self._initialize_parameters()
    
    def _initialize_parameters(self) -> None:
        """Initialize parameter predictors with reasonable biological ranges."""
        # Initialize beta predictor to produce values in typical splicing rate range
        with torch.no_grad():
            # Target initial beta values around 0.5-1.0
            self.beta_predictor.network[-1].bias.fill_(
                torch.log(torch.expm1(torch.tensor(0.8)))
            )
            
            # Target initial gamma values around 0.2-0.5
            self.gamma_predictor.network[-1].bias.fill_(
                torch.log(torch.expm1(torch.tensor(0.3)))
            )
            
            # Target initial time values around 0.5
            self.time_predictor.network[-1].bias.fill_(
                torch.log(torch.expm1(torch.tensor(0.5)))
            )
    
    def forward(
        self,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict ODE parameters from input features.
        
        Parameters
        ----------
        features : torch.Tensor
            Input features of shape (batch_size, input_dim).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'beta': Splicing rates of shape (batch_size, n_genes)
            - 'gamma': Degradation rates of shape (batch_size, n_genes) 
            - 'time': Cell times of shape (batch_size, 1)
        """
        # Predict raw parameters
        beta_raw = self.beta_predictor(features)
        gamma_raw = self.gamma_predictor(features)
        time_raw = self.time_predictor(features)
        
        # Apply softplus to ensure positivity and biological ranges
        beta = torch.clamp(
            F.softplus(beta_raw),
            min=self.config.ode.init_beta_range[0],
            max=self.config.ode.init_beta_range[1]
        )
        
        gamma = torch.clamp(
            F.softplus(gamma_raw),
            min=self.config.ode.init_gamma_range[0], 
            max=self.config.ode.init_gamma_range[1]
        )
        
        time = torch.clamp(
            F.softplus(time_raw),
            min=self.config.ode.init_time_range[0],
            max=self.config.ode.init_time_range[1]
        )
        
        return {
            'beta': beta,
            'gamma': gamma,
            'time': time
        }


class VelocityODE(nn.Module):
    """
    RNA velocity ODE system with regulatory transcription rates.
    
    This module defines the ODE system:
    du/dt = α(s) - β * u
    ds/dt = β * u - γ * s
    
    where α(s) comes from the regulatory network.
    
    Parameters
    ----------
    n_genes : int
        Number of genes.
    regulatory_network : RegulatoryNetwork
        Pre-configured regulatory network module.
    """
    
    def __init__(
        self,
        n_genes: int,
        regulatory_network
    ):
        super().__init__()
        self.n_genes = n_genes
        self.regulatory_network = regulatory_network
        
        # Store ODE parameters (set during forward pass)
        self.beta = None
        self.gamma = None
    
    def set_parameters(
        self,
        beta: torch.Tensor,
        gamma: torch.Tensor
    ) -> None:
        """
        Set ODE parameters for current batch.
        
        Parameters
        ----------
        beta : torch.Tensor
            Splicing rates of shape (batch_size, n_genes).
        gamma : torch.Tensor
            Degradation rates of shape (batch_size, n_genes).
        """
        self.beta = beta
        self.gamma = gamma
    
    def forward(
        self,
        t: Union[float, torch.Tensor],
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity vector dy/dt.
        
        Parameters
        ----------
        t : Union[float, torch.Tensor]
            Current time (often unused in autonomous systems).
        y : torch.Tensor
            State vector of shape (batch_size, 2*n_genes) where
            y = [u, s] (concatenated unspliced and spliced).
            
        Returns
        -------
        torch.Tensor
            Velocity vector dy/dt of same shape as y.
        """
        if self.beta is None or self.gamma is None:
            raise RuntimeError("ODE parameters must be set before forward pass.")
        
        batch_size = y.shape[0]
        
        # Split state vector into unspliced and spliced
        u = y[:, :self.n_genes]
        s = y[:, self.n_genes:]
        
        # Compute transcription rates from regulatory network
        alpha = self.regulatory_network(s)
        
        # ODE system
        du_dt = alpha - self.beta * u
        ds_dt = self.beta * u - self.gamma * s
        
        # Concatenate derivatives
        dy_dt = torch.cat([du_dt, ds_dt], dim=1)
        
        return dy_dt
    
    def validate_setup(self) -> None:
        """
        Validate that the ODE system is properly configured.
        
        Raises
        ------
        RuntimeError
            If the regulatory network or parameters are not properly set.
        """
        if self.regulatory_network is None:
            raise RuntimeError("Regulatory network not set. Call set_regulatory_network() first.")
        
        if self.beta is None or self.gamma is None:
            raise RuntimeError("ODE parameters not set. Call set_parameters() first.")
        
        # Validate parameter shapes
        if self.beta.shape[-1] != self.n_genes:
            raise RuntimeError(f"Beta parameter shape {self.beta.shape} does not match n_genes={self.n_genes}")
        
        if self.gamma.shape[-1] != self.n_genes:
            raise RuntimeError(f"Gamma parameter shape {self.gamma.shape} does not match n_genes={self.n_genes}")
    
    def get_velocity_components(
        self,
        spliced: torch.Tensor,
        unspliced: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get individual velocity components for analysis.
        
        Parameters
        ----------
        spliced : torch.Tensor
            Spliced RNA counts of shape (batch_size, n_genes).
        unspliced : torch.Tensor
            Unspliced RNA counts of shape (batch_size, n_genes).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing velocity components:
            - 'alpha': Transcription rates
            - 'beta_u': Beta * unspliced term
            - 'gamma_s': Gamma * spliced term
            - 'du_dt': Unspliced derivatives
            - 'ds_dt': Spliced derivatives
        """
        if self.beta is None or self.gamma is None:
            raise RuntimeError("ODE parameters must be set before computing components.")
        
        # Compute transcription rates
        alpha = self.regulatory_network(spliced)
        
        # Compute individual terms
        beta_u = self.beta * unspliced
        gamma_s = self.gamma * spliced
        
        # Compute derivatives
        du_dt = alpha - beta_u
        ds_dt = beta_u - gamma_s
        
        return {
            'alpha': alpha,
            'beta_u': beta_u,
            'gamma_s': gamma_s,
            'du_dt': du_dt,
            'ds_dt': ds_dt
        }


class ODESolver:
    """
    Wrapper for TorchODE solvers with Tangelo-specific configurations.
    
    This class provides a convenient interface for solving the RNA velocity
    ODE system using various numerical integration methods.
    
    Parameters
    ----------
    config : TangeloConfig
        Configuration object containing ODE solver parameters.
    """
    
    def __init__(self, config):
        self.config = config
        self.solver_name = config.ode.solver
        self.rtol = config.ode.rtol
        self.atol = config.ode.atol
        self.max_steps = config.ode.max_steps
    
    def solve(
        self,
        ode_system: VelocityODE,
        y0: torch.Tensor,
        t_span: Tuple[float, float],
        ode_params: Dict[str, torch.Tensor],
        n_time_points: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Solve the ODE system for given initial conditions.
        
        Parameters
        ----------
        ode_system : VelocityODE
            The ODE system to solve.
        y0 : torch.Tensor
            Initial conditions of shape (batch_size, 2*n_genes).
        t_span : Tuple[float, float]
            Start and end times for integration.
        ode_params : Dict[str, torch.Tensor]
            ODE parameters (beta, gamma, time).
        n_time_points : int, optional
            Number of time points to evaluate. Uses config default if None.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'solution': Full solution tensor
            - 'final_state': Final state at end time
            - 'times': Time points used
        """
        batch_size = y0.shape[0]
        n_time_points = n_time_points or self.config.ode.n_time_points
        
        # Set ODE parameters
        ode_system.set_parameters(
            beta=ode_params['beta'],
            gamma=ode_params['gamma']
        )
        
        # Use cell-specific end times if provided
        if 'time' in ode_params:
            t_end = ode_params['time'].squeeze(-1)  # (batch_size,)
        else:
            t_end = torch.full((batch_size,), t_span[1], device=y0.device)
        
        t_start = torch.full((batch_size,), t_span[0], device=y0.device)
        
        # Create evaluation times
        t_eval = torch.linspace(
            t_span[0], t_span[1], n_time_points, device=y0.device
        ).expand(batch_size, -1)
        
        # Create ODE term
        term = to.ODETerm(ode_system)
        
        # Select solver method
        if self.solver_name.lower() == "dopri5":
            step_method = to.Dopri5(term=term)
        elif self.solver_name.lower() == "tsit5":
            step_method = to.Tsit5(term=term)
        elif self.solver_name.lower() == "euler":
            step_method = to.Euler(term=term)
        else:
            raise ValueError(f"Unsupported solver: {self.solver_name}")
        
        # Create step size controller
        controller = to.PIDController(
            term=term,
            atol=self.atol,
            rtol=self.rtol,
        )
        
        # Create adjoint solver for gradient computation
        solver = to.AutoDiffAdjoint(
            step_method, 
            controller,
            max_steps=self.max_steps
        )
        
        # Create initial value problem
        problem = to.InitialValueProblem(
            y0=y0,
            t_start=t_start,
            t_end=t_end,
            t_eval=t_eval
        )
        
        # Solve the ODE
        solution = solver.solve(problem, dt0=torch.ones(1, device=y0.device) * 0.01)
        
        return {
            'solution': solution.ys,  # (time_points, batch_size, 2*n_genes)
            'final_state': solution.ys[-1],  # (batch_size, 2*n_genes)
            'times': solution.ts,  # (time_points, batch_size)
            'status': solution.status
        }
    
    def solve_batch_sequential(
        self,
        ode_system: VelocityODE,
        y0: torch.Tensor,
        t_span: Tuple[float, float],
        ode_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Solve ODE for each sample in batch sequentially (memory efficient).
        
        This method processes each sample individually to handle very large
        batches or when samples have different time spans.
        
        Parameters
        ----------
        ode_system : VelocityODE
            The ODE system to solve.
        y0 : torch.Tensor
            Initial conditions of shape (batch_size, 2*n_genes).
        t_span : Tuple[float, float]
            Start and end times for integration.
        ode_params : Dict[str, torch.Tensor]
            ODE parameters (beta, gamma, time).
            
        Returns
        -------
        torch.Tensor
            Final states of shape (batch_size, 2*n_genes).
        """
        batch_size = y0.shape[0]
        final_states = []
        
        for i in range(batch_size):
            # Extract parameters for single sample
            sample_params = {
                key: val[i:i+1] for key, val in ode_params.items()
            }
            sample_y0 = y0[i:i+1]
            
            # Solve for single sample
            result = self.solve(
                ode_system=ode_system,
                y0=sample_y0,
                t_span=t_span,
                ode_params=sample_params,
                n_time_points=2  # Only need start and end
            )
            
            final_states.append(result['final_state'])
        
        return torch.cat(final_states, dim=0)
    
    def validate_solver_config(self) -> None:
        """
        Validate solver configuration parameters.
        
        Raises
        ------
        ValueError
            If solver configuration parameters are invalid.
        """
        # Check solver name
        valid_solvers = ["dopri5", "tsit5", "euler"]
        if self.solver_name.lower() not in valid_solvers:
            raise ValueError(f"Invalid solver '{self.solver_name}'. Must be one of {valid_solvers}")
        
        # Check tolerances
        if self.atol <= 0 or self.rtol <= 0:
            raise ValueError(f"Tolerances must be positive. Got atol={self.atol}, rtol={self.rtol}")
        
        # Check max steps
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive. Got {self.max_steps}")
    
    def get_solver_info(self) -> Dict[str, Any]:
        """
        Get information about the solver configuration.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing solver configuration details.
        """
        return {
            'solver_name': self.solver_name,
            'rtol': self.rtol,
            'atol': self.atol,
            'max_steps': self.max_steps,
            'is_adaptive': self.solver_name.lower() in ["dopri5", "tsit5"]
        }