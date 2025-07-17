"""
Mixed Variational Posterior for Bayesian Neural Networks

Implementation of the MixedVariationalDistributionalCritic based on:
"Alpha-divergence minimization with mixed variational posterior for
Bayesian neural networks and its robustness against adversarial examples"
by Xiao Liu and Shiliang Sun, Neurocomputing 2021

This implementation provides a standalone Bayesian neural network module
that can be integrated into various reinforcement learning or supervised
learning frameworks.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class MixedVariationalDistributionalCritic(nn.Module):
    """
    Bayesian Neural Network with Mixed Variational Posterior Distribution

    This implementation replaces traditional Gaussian variational posteriors with
    mixture distributions to achieve better uncertainty estimation and robustness.

    Key Features:
    - Mixed exponential family distributions as variational posterior
    - Gumbel-Softmax reparameterization for discrete mixture weights
    - Separate aleatoric and epistemic uncertainty estimation
    - Compatible with α-divergence minimization framework

    Args:
        input_dim (int): Input dimension (e.g., state_dim + action_dim)
        output_dim (int): Output dimension (typically 1 for Q-values)
        n_components (int): Number of mixture components (default: 2)
        mc_samples (int): Monte Carlo samples for training (default: 10)
        temperature (float): Gumbel-Softmax temperature (default: 1.0)
        hidden_dims (list): Hidden layer dimensions (default: [256, 256])

    References:
        Liu, X., & Sun, S. (2021). Alpha-divergence minimization with mixed
        variational posterior for Bayesian neural networks and its robustness
        against adversarial examples. Neurocomputing, 423, 427-434.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 n_components: int = 2,
                 mc_samples: int = 10,
                 temperature: float = 1.0,
                 hidden_dims: list = [256, 256]):
        super(MixedVariationalDistributionalCritic, self).__init__()

        # Validate inputs
        assert n_components >= 1, "Number of components must be at least 1"
        assert mc_samples >= 1, "MC samples must be at least 1"
        assert temperature > 0, "Temperature must be positive"
        assert len(hidden_dims) >= 1, "At least one hidden layer required"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components
        self.mc_samples = mc_samples
        self.temperature = temperature
        self.hidden_dims = hidden_dims

        # Build network layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        # Mixed variational posterior parameters
        # Each component outputs 'output_dim' values
        self.component_means = nn.Linear(prev_dim, n_components * output_dim)
        self.component_log_vars = nn.Linear(prev_dim, n_components * output_dim)

        # Categorical distribution parameters (mixture weights)
        self.mixture_logits = nn.Linear(prev_dim, n_components)

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize network parameters with reasonable defaults"""
        # Xavier initialization for hidden layers
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Initialize output layers
        nn.init.xavier_uniform_(self.component_means.weight)
        nn.init.zeros_(self.component_means.bias)

        nn.init.xavier_uniform_(self.component_log_vars.weight)
        nn.init.constant_(self.component_log_vars.bias, -2.0)  # Small initial variance

        # Initialize mixture to be uniform
        nn.init.zeros_(self.mixture_logits.weight)
        nn.init.zeros_(self.mixture_logits.bias)

    def gumbel_softmax_reparameterization(self,
                                          logits: torch.Tensor,
                                          temperature: float,
                                          hard: bool = False) -> torch.Tensor:
        """
        Gumbel-Softmax reparameterization trick from Equation (5) in the paper:
        x := f(π,ε) = softmax((log(π) + g)/τ)

        Args:
            logits: Categorical distribution logits [batch_size, n_components]
            temperature: Softmax temperature τ
            hard: Whether to use straight-through estimator

        Returns:
            Sampled categorical probabilities [batch_size, n_components]
        """
        # Sample Gumbel noise: g = -log(-log(u)) where u ~ Uniform(0,1)
        eps = 1e-20
        uniform_samples = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform_samples + eps) + eps)

        # Apply Gumbel-Softmax: softmax((log(π) + g)/τ)
        y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)

        if hard:
            # Straight-through estimator for discrete sampling
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y

        return y

    def sample_mixed_variational_posterior(self,
                                           means: torch.Tensor,
                                           log_vars: torch.Tensor,
                                           mixture_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from mixed variational posterior: q(x) = Σᵢ bᵢ qᵢ(x)

        Args:
            means: Component means [batch_size, n_components, output_dim]
            log_vars: Component log variances [batch_size, n_components, output_dim]
            mixture_weights: Mixture weights [batch_size, n_components]

        Returns:
            q_sample: Weighted sample [batch_size, output_dim]
            component_samples: Individual component samples [batch_size, n_components, output_dim]
        """
        variances = F.softplus(log_vars)  # Ensure positive variance

        # Sample from each Gaussian component: xᵢ ~ N(μᵢ, σᵢ²)
        eps = torch.randn_like(means)
        component_samples = means + torch.sqrt(variances) * eps

        # Weighted combination: x = Σᵢ bᵢ xᵢ
        # mixture_weights: [batch_size, n_components] -> [batch_size, n_components, 1]
        weights_expanded = mixture_weights.unsqueeze(-1)
        q_sample = torch.sum(weights_expanded * component_samples, dim=1)

        return q_sample, component_samples

    def forward(self,
                x: torch.Tensor,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Forward pass through the mixed variational network

        Args:
            x: Input tensor [batch_size, input_dim]
            training: Whether in training mode (enables MC sampling)

        Returns:
            mean_output: Expected output [batch_size, output_dim]
            variance_output: Output variance [batch_size, output_dim]
            samples: MC samples if training [mc_samples, batch_size, output_dim] or None
            info: Dictionary with posterior information
        """
        batch_size = x.shape[0]

        # Forward pass through hidden layers
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))

        # Get mixture parameters
        means_flat = self.component_means(h)  # [batch_size, n_components * output_dim]
        log_vars_flat = self.component_log_vars(h)  # [batch_size, n_components * output_dim]
        mixture_logits = self.mixture_logits(h)  # [batch_size, n_components]

        # Reshape to separate components
        means = means_flat.view(batch_size, self.n_components, self.output_dim)
        log_vars = log_vars_flat.view(batch_size, self.n_components, self.output_dim)

        if training and self.mc_samples > 1:
            return self._forward_training(means, log_vars, mixture_logits)
        else:
            return self._forward_evaluation(means, log_vars, mixture_logits)

    def _forward_training(self, means, log_vars, mixture_logits):
        """Forward pass during training with Monte Carlo sampling"""
        q_samples = []
        mixture_weights_samples = []

        for _ in range(self.mc_samples):
            # Sample mixture weights using Gumbel-Softmax
            mixture_weights = self.gumbel_softmax_reparameterization(
                mixture_logits, self.temperature
            )
            mixture_weights_samples.append(mixture_weights)

            # Sample from mixed variational posterior
            q_sample, _ = self.sample_mixed_variational_posterior(
                means, log_vars, mixture_weights
            )
            q_samples.append(q_sample)

        # Stack samples and compute statistics
        q_samples = torch.stack(q_samples)  # [mc_samples, batch_size, output_dim]
        mixture_weights_samples = torch.stack(mixture_weights_samples)

        mean_output = torch.mean(q_samples, dim=0)
        avg_mixture_weights = torch.mean(mixture_weights_samples, dim=0)

        # Compute mixture variance: Var[X] = E[Var[X|Z]] + Var[E[X|Z]]
        variances = F.softplus(log_vars)

        # E[X|Z] for each component
        weights_expanded = avg_mixture_weights.unsqueeze(-1)  # [batch_size, n_components, 1]
        mixture_mean = torch.sum(weights_expanded * means, dim=1)  # [batch_size, output_dim]

        # Var[X] = E[Var[X|Z]] + Var[E[X|Z]]
        mean_of_vars = torch.sum(weights_expanded * variances, dim=1)  # E[Var[X|Z]]
        var_of_means = torch.sum(weights_expanded * (means - mixture_mean.unsqueeze(1)) ** 2, dim=1)  # Var[E[X|Z]]
        mixture_var = mean_of_vars + var_of_means
        mixture_var = torch.clamp(mixture_var, min=1e-8)

        info = {
            'means': means,
            'variances': variances,
            'mixture_weights': avg_mixture_weights,
            'mixture_logits': mixture_logits,
            'mc_samples': self.mc_samples
        }

        return mean_output, mixture_var, q_samples, info

    def _forward_evaluation(self, means, log_vars, mixture_logits):
        """Forward pass during evaluation (deterministic)"""
        mixture_weights = F.softmax(mixture_logits, dim=-1)
        variances = F.softplus(log_vars)

        # Compute mixture statistics
        weights_expanded = mixture_weights.unsqueeze(-1)  # [batch_size, n_components, 1]
        mixture_mean = torch.sum(weights_expanded * means, dim=1)  # [batch_size, output_dim]

        # Mixture variance
        mean_of_vars = torch.sum(weights_expanded * variances, dim=1)
        var_of_means = torch.sum(weights_expanded * (means - mixture_mean.unsqueeze(1)) ** 2, dim=1)
        mixture_var = mean_of_vars + var_of_means
        mixture_var = torch.clamp(mixture_var, min=1e-8)

        info = {
            'means': means,
            'variances': variances,
            'mixture_weights': mixture_weights,
            'mixture_logits': mixture_logits,
            'mc_samples': 1
        }

        return mixture_mean, mixture_var, None, info

    def get_uncertainty(self,
                        x: torch.Tensor,
                        n_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate uncertainty using the mixed variational posterior

        Decomposes total uncertainty into:
        - Epistemic uncertainty: Uncertainty in model parameters (reducible with more data)
        - Aleatoric uncertainty: Inherent noise in observations (irreducible)

        Args:
            x: Input tensor [batch_size, input_dim]
            n_samples: Number of samples for uncertainty estimation

        Returns:
            mean_prediction: Expected output [batch_size, output_dim]
            epistemic_uncertainty: Model uncertainty [batch_size, output_dim]
            aleatoric_uncertainty: Data uncertainty [batch_size, output_dim]
            total_uncertainty: Combined uncertainty [batch_size, output_dim]
        """
        original_mode = self.training
        self.eval()  # Set to evaluation mode

        batch_size = x.shape[0]

        with torch.no_grad():
            # Forward pass through hidden layers
            h = x
            for layer in self.layers:
                h = F.relu(layer(h))

            means_flat = self.component_means(h)
            log_vars_flat = self.component_log_vars(h)
            mixture_logits = self.mixture_logits(h)

            means = means_flat.view(batch_size, self.n_components, self.output_dim)
            log_vars = log_vars_flat.view(batch_size, self.n_components, self.output_dim)

            # Sample multiple times to estimate uncertainty
            q_samples = []
            mixture_weights_samples = []

            for _ in range(n_samples):
                mixture_weights = self.gumbel_softmax_reparameterization(
                    mixture_logits, self.temperature
                )
                mixture_weights_samples.append(mixture_weights)

                q_sample, _ = self.sample_mixed_variational_posterior(
                    means, log_vars, mixture_weights
                )
                q_samples.append(q_sample)

            q_samples = torch.stack(q_samples)  # [n_samples, batch_size, output_dim]
            mixture_weights_samples = torch.stack(mixture_weights_samples)

            # Epistemic uncertainty: Uncertainty in the mean prediction
            epistemic_uncertainty = torch.std(q_samples, dim=0)

            # Aleatoric uncertainty: Expected inherent noise
            avg_mixture_weights = torch.mean(mixture_weights_samples, dim=0)
            variances = F.softplus(log_vars)
            weights_expanded = avg_mixture_weights.unsqueeze(-1)
            aleatoric_uncertainty = torch.sqrt(
                torch.sum(weights_expanded * variances, dim=1)
            )

            # Total uncertainty
            total_uncertainty = torch.sqrt(epistemic_uncertainty ** 2 + aleatoric_uncertainty ** 2)

            # Mean prediction
            mean_prediction = torch.mean(q_samples, dim=0)

        # Restore original training mode
        self.train(original_mode)

        return mean_prediction, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty

    def compute_log_prob(self,
                         q_samples: torch.Tensor,
                         means: torch.Tensor,
                         log_vars: torch.Tensor,
                         mixture_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of samples under the mixed variational posterior
        Used for α-divergence computation in training

        Args:
            q_samples: Samples [mc_samples, batch_size, output_dim]
            means: Component means [batch_size, n_components, output_dim]
            log_vars: Component log variances [batch_size, n_components, output_dim]
            mixture_weights: Mixture weights [batch_size, n_components]

        Returns:
            log_probs: Log probabilities [mc_samples, batch_size]
        """
        variances = F.softplus(log_vars)
        mc_samples, batch_size, output_dim = q_samples.shape

        # Reshape for broadcasting
        q_expanded = q_samples.unsqueeze(2)  # [mc_samples, batch_size, 1, output_dim]
        means_expanded = means.unsqueeze(0)  # [1, batch_size, n_components, output_dim]
        vars_expanded = variances.unsqueeze(0)  # [1, batch_size, n_components, output_dim]
        weights_expanded = mixture_weights.unsqueeze(0).unsqueeze(-1)  # [1, batch_size, n_components, 1]

        # Gaussian log probabilities for each component
        log_probs = -0.5 * torch.sum(torch.log(2 * np.pi * vars_expanded), dim=-1) - \
                    0.5 * torch.sum((q_expanded - means_expanded) ** 2 / vars_expanded, dim=-1)
        # log_probs: [mc_samples, batch_size, n_components]

        # Weighted mixture probabilities
        weighted_probs = weights_expanded.squeeze(-1) * torch.exp(log_probs)
        mixture_probs = torch.sum(weighted_probs, dim=-1)  # [mc_samples, batch_size]

        return torch.log(mixture_probs + 1e-8)

    def get_model_info(self) -> Dict:
        """Get model architecture and complexity information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'architecture': 'Mixed Variational Bayesian Network',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'n_components': self.n_components,
            'mc_samples': self.mc_samples,
            'temperature': self.temperature,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


def create_regression_network(input_dim: int,
                              n_components: int = 2,
                              mc_samples: int = 10) -> MixedVariationalDistributionalCritic:
    """
    Factory function to create a regression network

    Args:
        input_dim: Input feature dimension
        n_components: Number of mixture components
        mc_samples: Monte Carlo samples for training

    Returns:
        Configured MixedVariationalDistributionalCritic for regression
    """
    return MixedVariationalDistributionalCritic(
        input_dim=input_dim,
        output_dim=1,
        n_components=n_components,
        mc_samples=mc_samples,
        hidden_dims=[256, 256]
    )


def create_classification_network(input_dim: int,
                                  num_classes: int,
                                  n_components: int = 2,
                                  mc_samples: int = 10) -> MixedVariationalDistributionalCritic:
    """
    Factory function to create a classification network

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        n_components: Number of mixture components
        mc_samples: Monte Carlo samples for training

    Returns:
        Configured MixedVariationalDistributionalCritic for classification
    """
    return MixedVariationalDistributionalCritic(
        input_dim=input_dim,
        output_dim=num_classes,
        n_components=n_components,
        mc_samples=mc_samples,
        hidden_dims=[512, 256]
    )


# Example usage
if __name__ == "__main__":
    print("🧠 Mixed Variational Bayesian Neural Network")
    print("=" * 50)

    # Example 1: Basic usage
    input_dim = 10
    model = MixedVariationalDistributionalCritic(
        input_dim=input_dim,
        output_dim=1,
        n_components=3,
        mc_samples=10
    )

    # Generate dummy data
    batch_size = 32
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    print("🔄 Forward pass:")
    mean, var, samples, info = model(x, training=True)
    print(f"   Mean shape: {mean.shape}")
    print(f"   Variance shape: {var.shape}")
    print(f"   MC samples shape: {samples.shape if samples is not None else None}")

    # Uncertainty estimation
    print("\n📊 Uncertainty estimation:")
    mean_pred, epistemic, aleatoric, total = model.get_uncertainty(x, n_samples=20)
    print(f"   Epistemic uncertainty: {epistemic.mean().item():.4f} ± {epistemic.std().item():.4f}")
    print(f"   Aleatoric uncertainty: {aleatoric.mean().item():.4f} ± {aleatoric.std().item():.4f}")
    print(f"   Total uncertainty: {total.mean().item():.4f} ± {total.std().item():.4f}")

    # Model info
    print(f"\n🔧 Model info:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    print("\n✅ Example completed successfully!")