"""
Basic usage example for Tangelo Velocity.

This example demonstrates how to use Tangelo Velocity for multi-modal
single-cell velocity estimation with spatial transcriptomics and ATAC-seq data.
"""

import numpy as np
import pandas as pd
import muon as mu
import scanpy as sc
import tangelo_velocity as tv

# Set up scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')


def load_example_data():
    """
    Load or create example multi-modal data.
    
    In practice, you would load your own MuData object with:
    adata = mu.read_h5mu("your_data.h5mu")
    """
    # This is a placeholder - replace with actual data loading
    print("Loading example multi-modal data...")
    print("(In practice, use: adata = mu.read_h5mu('your_data.h5mu'))")
    
    # For demonstration, we'll assume you have a properly formatted MuData object
    # with the required structure:
    # - obs: 'x_pixel', 'y_pixel', 'x_position', 'y_position'  
    # - modalities: 'rna' (with 'spliced', 'unspliced', 'open_chromatin' layers)
    #               'atac' (with 'counts', 'X_tfidf' layers)
    
    return None  # Replace with actual data loading


def basic_velocity_estimation(adata):
    """Demonstrate basic velocity estimation."""
    
    print("\n=== Basic Velocity Estimation ===")
    
    # Simple one-line velocity estimation (Stage 3 by default)
    tv.estimate_velocity(adata)
    
    # Check results
    print(f"Velocity computed: {'velocity' in adata['rna'].layers}")
    print(f"Velocity shape: {adata['rna'].layers['velocity'].shape}")


def advanced_velocity_estimation(adata):
    """Demonstrate advanced API usage."""
    
    print("\n=== Advanced Velocity Estimation ===")
    
    # Create custom configuration
    config = tv.TangeloConfig(
        development_stage=3,
        graph=tv.config.GraphConfig(
            n_neighbors_spatial=8,
            n_neighbors_expression=15,
            use_node2vec=False,
        ),
        encoder=tv.config.EncoderConfig(
            latent_dim=64,
            hidden_dims=(512, 256, 128),
            fusion_method="attention",
        ),
        training=tv.config.TrainingConfig(
            n_epochs=100,
            learning_rate=1e-3,
            batch_size=512,
        )
    )
    
    # Initialize model with custom config
    model = tv.TangeloVelocity(config=config)
    
    # Fit model
    model.fit(adata)
    
    # Compute velocity graph and embedding
    model.compute_velocity_graph(adata, n_neighbors=30)
    model.compute_velocity_embedding(adata, basis="umap")
    
    print("Advanced velocity estimation completed!")


def extract_model_components(model):
    """Demonstrate extraction of model components."""
    
    print("\n=== Model Component Extraction ===")
    
    # Get latent representations
    latent_reps = model.get_latent_representations()
    print(f"Spatial latent shape: {latent_reps['spatial_latent'].shape}")
    print(f"Expression latent shape: {latent_reps['expression_latent'].shape}")
    print(f"Combined latent shape: {latent_reps['combined_latent'].shape}")
    
    # Get ODE parameters
    ode_params = model.get_ode_parameters()
    print(f"Splicing rates (beta) shape: {ode_params['beta'].shape}")
    print(f"Degradation rates (gamma) shape: {ode_params['gamma'].shape}")
    print(f"Cell times shape: {ode_params['time'].shape}")
    
    # Get interaction network
    interaction_matrix = model.get_interaction_network()
    print(f"Interaction network shape: {interaction_matrix.shape}")


def stage_comparison(adata):
    """Compare velocity estimates across different stages."""
    
    print("\n=== Stage Comparison ===")
    
    # Compare multiple stages
    results = tv.compare_stages(
        adata.copy(),  # Use copy to avoid modifying original
        stages=(1, 2, 3),
        n_epochs=50  # Reduced for demonstration
    )
    
    # Analyze results
    for stage, result in results.items():
        velocity_norm = np.linalg.norm(result['rna'].layers['velocity'], axis=1)
        print(f"Stage {stage} - Mean velocity magnitude: {velocity_norm.mean():.4f}")


def visualization_examples(adata):
    """Demonstrate visualization capabilities."""
    
    print("\n=== Visualization Examples ===")
    
    # Velocity embedding plot
    tv.plotting.plot_velocity_embedding(
        adata,
        basis="umap",
        color="leiden",  # Assuming you have leiden clustering
        save="velocity_embedding.pdf"
    )
    
    # Spatial velocity plot
    tv.plotting.plot_spatial_velocity(
        adata,
        color="leiden",
        save="spatial_velocity.pdf"
    )
    
    # Plot specific genes spatially
    tv.plotting.plot_spatial_gene_expression(
        adata,
        genes=["gene1", "gene2"],  # Replace with actual gene names
        save="spatial_genes.pdf"
    )
    
    print("Visualization plots saved!")


def perturbation_analysis_example(model, adata):
    """Demonstrate perturbation analysis."""
    
    print("\n=== Perturbation Analysis ===")
    
    # Initialize perturbation analysis
    perturbation = tv.analysis.PerturbationAnalysis(model)
    
    # Simulate gene knockdown
    target_gene = "target_gene"  # Replace with actual gene name
    knockdown_result = perturbation.simulate_knockdown(
        gene=target_gene,
        knockdown_strength=0.8
    )
    
    # Analyze fate changes
    fate_changes = perturbation.analyze_fate_changes(knockdown_result)
    
    print(f"Perturbation analysis completed for {target_gene}")
    print(f"Number of cells with fate changes: {len(fate_changes)}")


def metrics_evaluation(adata):
    """Demonstrate velocity metrics evaluation."""
    
    print("\n=== Metrics Evaluation ===")
    
    # Initialize metrics
    metrics = tv.analysis.VelocityMetrics(adata)
    
    # Compute various metrics
    summary = metrics.summary()
    
    print("Velocity Quality Metrics:")
    for metric, value in summary.items():
        print(f"  {metric}: {value:.4f}")


def main():
    """Main example workflow."""
    
    print("Tangelo Velocity - Basic Usage Example")
    print("=" * 50)
    
    # Load data
    adata = load_example_data()
    
    if adata is None:
        print("\nPlease replace load_example_data() with actual data loading!")
        print("Expected format:")
        print("  - MuData object with 'rna' and 'atac' modalities")
        print("  - Spatial coordinates: 'x_pixel', 'y_pixel' in obs")
        print("  - RNA layers: 'spliced', 'unspliced', 'open_chromatin'")
        print("  - ATAC layers: 'counts', 'X_tfidf'")
        return
    
    # Basic usage
    basic_velocity_estimation(adata.copy())
    
    # Advanced usage
    model = advanced_velocity_estimation(adata.copy())
    
    # Extract model components
    extract_model_components(model)
    
    # Stage comparison
    stage_comparison(adata)
    
    # Visualization
    visualization_examples(adata)
    
    # Perturbation analysis
    perturbation_analysis_example(model, adata)
    
    # Metrics evaluation
    metrics_evaluation(adata)
    
    print("\n" + "=" * 50)
    print("Example completed! Check the generated plots and results.")


if __name__ == "__main__":
    main()