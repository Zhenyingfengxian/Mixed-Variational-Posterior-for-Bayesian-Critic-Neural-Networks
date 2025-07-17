# Mixed Variational Posterior for Bayesian Neural Networks

A PyTorch implementation of mixed variational posterior distributions for Bayesian neural networks.

## 🔥 Key Features
- **Mixed exponential family distributions** instead of simple Gaussian posteriors
- **Gumbel-Softmax reparameterization** for discrete mixture weights
- **Uncertainty decomposition** into epistemic and aleatoric components
- **α-divergence compatible** training framework
- **General purpose** - works for RL/SRL

## 🚀 Quick Start
```python
from mixed_variational_critic import MixedVariationalDistributionalCritic

# Create model
model = MixedVariationalDistributionalCritic(
    input_dim=10, 
    output_dim=1, 
    n_components=2
)

# Get predictions with uncertainty
mean, var, _, _ = model(x, training=True)
mean_pred, epistemic, aleatoric, total = model.get_uncertainty(x)
