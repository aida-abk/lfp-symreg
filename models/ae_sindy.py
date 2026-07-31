"""Deep delay autoencoder with a SINDy latent model, in PyTorch.

This is a compact reimplementation of the architecture in Bakarji, Champion,
Kutz & Brunton, "Discovering governing equations from partial measurements
with deep delay autoencoders" (Proc. R. Soc. A 479:20230422, 2023), whose
reference code is TensorFlow (github.com/josephbakarji/deep-delay-autoencoder,
package ``aesindy``).

Pipeline, matching the paper:

    scalar trace -> Hankel/delay embedding (dimension q)
                 -> encoder -> latent state z (dimension d)
                 -> SINDy polynomial ODE in z
                 -> decoder -> back to delay space

Losses implemented here, with the reference implementation's names:

    reconstruction   ||x - decoder(encoder(x))||^2
    sindy_z          ||dz/dt - Theta(z) Xi||^2      where dz/dt = J_enc(x) xdot
    sindy_x          ||xdot - J_dec(z) Theta(z) Xi||^2
    regularization   |Xi|_1

Deliberately omitted: the reference's RK4 ``integral`` loss and its ``x0``
loss. The integral loss is where most of that implementation's numerical
fragility lives -- it clamps at +/-500 -- and it is an addition to, not the
core of, the method. Omitting it is a documented deviation, not an oversight.

Jacobians are propagated analytically through the network rather than taken
with autograd. For an MLP the derivative of the activations can be carried
forward alongside the activations themselves in a single pass, which is both
faster and simpler than a full Jacobian, and is what the reference
implementation does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np

try:
  import torch
  from torch import nn
except ImportError as exc:  # pragma: no cover - environment dependent
  raise ImportError(
    "PyTorch is required for the deep delay autoencoder. Install it with "
    "`pip install torch`, or run the PySINDy-only scripts instead."
  ) from exc


def polynomial_exponents(latent_dim: int, poly_order: int) -> list[tuple[int, ...]]:
  """Enumerate polynomial library exponents, constant term first.

  Args:
    latent_dim: Number of latent coordinates. Unitless count.
    poly_order: Maximum total degree. Unitless.

  Returns:
    Exponent tuples ordered by increasing total degree, each of length
    ``latent_dim``. The first entry is the all-zero constant term.
  """
  exponents = [tuple([0] * latent_dim)]
  for degree in range(1, poly_order + 1):
    for combination in combinations_with_replacement(range(latent_dim), degree):
      exponent = [0] * latent_dim
      for index in combination:
        exponent[index] += 1
      exponents.append(tuple(exponent))
  return exponents


def feature_names(exponents: list[tuple[int, ...]]) -> list[str]:
  """Return readable names for polynomial library features.

  Args:
    exponents: Exponent tuples from :func:`polynomial_exponents`.

  Returns:
    Names such as ``1``, ``z0``, ``z0 z1``, ``z0^2``.
  """
  names = []
  for exponent in exponents:
    if not any(exponent):
      names.append("1")
      continue
    parts = []
    for index, power in enumerate(exponent):
      if power == 1:
        parts.append(f"z{index}")
      elif power > 1:
        parts.append(f"z{index}^{power}")
    names.append(" ".join(parts))
  return names


def polynomial_library_numpy(
  states: np.ndarray, exponents: list[tuple[int, ...]]
) -> np.ndarray:
  """Evaluate the polynomial library on numpy states.

  Used after training, when the latent ODE is integrated with SciPy.

  Args:
    states: Latent states with shape ``(samples, latent_dim)``.
    exponents: Exponent tuples from :func:`polynomial_exponents`.

  Returns:
    Feature matrix with shape ``(samples, n_features)``.
  """
  values = np.atleast_2d(np.asarray(states, dtype=float))
  columns = []
  for exponent in exponents:
    column = np.ones(values.shape[0], dtype=float)
    for index, power in enumerate(exponent):
      if power:
        column = column * values[:, index] ** power
    columns.append(column)
  return np.stack(columns, axis=1)


@dataclass
class AESINDyConfig:
  """Hyperparameters for one deep delay autoencoder.

  Defaults follow ``testcases/lorenzww_basic.py`` and ``default_params.py``
  in the reference repository, which is its only experimental-data example.

  Attributes:
    input_dim: Hankel embedding dimension. The reference uses 80.
    latent_dim: Latent state dimension the SINDy model is written in.
    poly_order: Maximum polynomial degree in the latent library.
    widths_ratios: Hidden layer widths as fractions of ``input_dim``.
    learning_rate: Adam step size.
    batch_size: Minibatch size in delay vectors.
    max_epochs: Maximum training epochs.
    patience: Epochs without validation improvement before stopping.
    loss_weight_rec: Weight on the reconstruction term.
    loss_weight_sindy_z: Weight on the latent-derivative term.
    loss_weight_sindy_x: Weight on the input-derivative term.
    loss_weight_regularization: Weight on the L1 penalty over ``Xi``.
    coefficient_threshold: Magnitude below which coefficients are masked out
      during refinement.
    threshold_frequency: Epoch interval between refinement passes.
    seed: Seed for parameter initialization and batching.
  """

  input_dim: int = 80
  latent_dim: int = 3
  poly_order: int = 2
  widths_ratios: tuple[float, ...] = (0.5, 0.25)
  learning_rate: float = 1e-3
  batch_size: int = 256
  max_epochs: int = 300
  patience: int = 20
  loss_weight_rec: float = 0.3
  loss_weight_sindy_z: float = 0.001
  loss_weight_sindy_x: float = 0.001
  loss_weight_regularization: float = 1e-5
  coefficient_threshold: float = 0.1
  threshold_frequency: int = 50
  seed: int = 0


class DerivativeMLP(nn.Module):
  """An ELU MLP that also propagates directional derivatives.

  A forward pass carries both the activations and the derivative of those
  activations along an input direction, so ``J(x) v`` is available without
  forming the Jacobian.

  Attributes:
    layers: The linear layers, applied with ELU between them.
  """

  def __init__(self, widths: list[int]) -> None:
    """Build the network.

    Args:
      widths: Layer sizes from input to output, inclusive.
    """
    super().__init__()
    self.layers = nn.ModuleList(
      nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Return the network output.

    Args:
      x: Input with shape ``(batch, in_dim)``.

    Returns:
      Output with shape ``(batch, out_dim)``.
    """
    for index, layer in enumerate(self.layers):
      x = layer(x)
      if index < len(self.layers) - 1:
        x = torch.nn.functional.elu(x)
    return x

  def forward_with_derivative(
    self, x: torch.Tensor, dx: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the output and its derivative along ``dx``.

    Args:
      x: Input with shape ``(batch, in_dim)``.
      dx: Input-space direction with the same shape.

    Returns:
      ``(output, d_output)``, where ``d_output`` is ``J(x) dx``.
    """
    for index, layer in enumerate(self.layers):
      x = layer(x)
      dx = dx @ layer.weight.T
      if index < len(self.layers) - 1:
        # d/du elu(u) is 1 for u > 0 and elu(u) + 1 otherwise.
        activated = torch.nn.functional.elu(x)
        slope = torch.where(x > 0, torch.ones_like(x), activated + 1.0)
        dx = dx * slope
        x = activated
    return x, dx


class AESINDy(nn.Module):
  """Autoencoder with a sparse polynomial ODE in the latent space.

  Attributes:
    config: Hyperparameters this model was built from.
    encoder: Maps delay vectors to latent states.
    decoder: Maps latent states back to delay vectors.
    coefficients: Latent ODE coefficients with shape ``(n_features, latent_dim)``.
    mask: Binary support mask with the same shape as ``coefficients``.
    exponents: Polynomial library exponents.
  """

  def __init__(self, config: AESINDyConfig) -> None:
    """Build encoder, decoder, and latent coefficient matrix.

    Args:
      config: Hyperparameters.
    """
    super().__init__()
    torch.manual_seed(config.seed)
    self.config = config
    hidden = [max(int(round(r * config.input_dim)), config.latent_dim)
              for r in config.widths_ratios]
    self.encoder = DerivativeMLP([config.input_dim, *hidden, config.latent_dim])
    self.decoder = DerivativeMLP([config.latent_dim, *hidden[::-1], config.input_dim])
    self.exponents = polynomial_exponents(config.latent_dim, config.poly_order)
    n_features = len(self.exponents)
    self.coefficients = nn.Parameter(
      torch.randn(n_features, config.latent_dim) * 0.1
    )
    self.register_buffer("mask", torch.ones(n_features, config.latent_dim))

  def library(self, z: torch.Tensor) -> torch.Tensor:
    """Evaluate the polynomial library on latent states.

    Args:
      z: Latent states with shape ``(batch, latent_dim)``.

    Returns:
      Feature matrix with shape ``(batch, n_features)``.
    """
    columns = []
    for exponent in self.exponents:
      column = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
      for index, power in enumerate(exponent):
        if power:
          column = column * z[:, index] ** power
      columns.append(column)
    return torch.stack(columns, dim=1)

  def latent_derivative(self, z: torch.Tensor) -> torch.Tensor:
    """Return the SINDy prediction of ``dz/dt``.

    Args:
      z: Latent states with shape ``(batch, latent_dim)``.

    Returns:
      Latent derivatives with shape ``(batch, latent_dim)``.
    """
    return self.library(z) @ (self.coefficients * self.mask)

  def losses(self, x: torch.Tensor, xdot: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute every loss term for one batch.

    Args:
      x: Delay vectors with shape ``(batch, input_dim)``, standardized.
      xdot: Their time derivatives with the same shape and scaling.

    Returns:
      Individual loss terms plus the weighted ``total``.
    """
    z, dz_encoded = self.encoder.forward_with_derivative(x, xdot)
    dz_sindy = self.latent_derivative(z)
    x_hat, dx_decoded = self.decoder.forward_with_derivative(z, dz_sindy)

    reconstruction = torch.mean((x - x_hat) ** 2)
    sindy_z = torch.mean((dz_encoded - dz_sindy) ** 2)
    sindy_x = torch.mean((xdot - dx_decoded) ** 2)
    regularization = torch.mean(torch.abs(self.coefficients * self.mask))

    config = self.config
    total = (
      config.loss_weight_rec * reconstruction
      + config.loss_weight_sindy_z * sindy_z
      + config.loss_weight_sindy_x * sindy_x
      + config.loss_weight_regularization * regularization
    )
    return {
      "reconstruction": reconstruction,
      "sindy_z": sindy_z,
      "sindy_x": sindy_x,
      "regularization": regularization,
      "total": total,
    }

  @torch.no_grad()
  def refine_mask(self) -> int:
    """Zero out coefficients below the configured threshold.

    This is the reference implementation's recursive feature elimination: once
    a coefficient is masked it stays masked, so the support only shrinks.

    Returns:
      The number of surviving coefficients.
    """
    keep = (self.coefficients * self.mask).abs() >= self.config.coefficient_threshold
    self.mask.mul_(keep.to(self.mask.dtype))
    return int(self.mask.sum().item())

  def equations(self) -> list[str]:
    """Return the latent ODE as readable strings."""
    names = feature_names(self.exponents)
    matrix = (self.coefficients * self.mask).detach().cpu().numpy()
    lines = []
    for state in range(matrix.shape[1]):
      terms = [
        f"{matrix[feature, state]:+.4g} {names[feature]}"
        for feature in range(matrix.shape[0])
        if abs(matrix[feature, state]) > 1e-12
      ]
      lines.append(f"dz{state}/dt = " + (" ".join(terms) if terms else "0"))
    return lines


@dataclass
class TrainingHistory:
  """Per-epoch training and validation losses.

  Attributes:
    train_total: Weighted training loss per epoch.
    validation_total: Weighted validation loss per epoch.
    active_terms: Surviving coefficient count per epoch.
  """

  train_total: list[float] = field(default_factory=list)
  validation_total: list[float] = field(default_factory=list)
  active_terms: list[int] = field(default_factory=list)


def train_ae_sindy(
  model: AESINDy,
  x_train: np.ndarray,
  xdot_train: np.ndarray,
  x_validation: np.ndarray,
  xdot_validation: np.ndarray,
  config: AESINDyConfig,
  verbose: bool = True,
) -> TrainingHistory:
  """Train the autoencoder with early stopping and periodic mask refinement.

  Args:
    model: Model to train in place.
    x_train: Standardized training delay vectors, shape ``(n, input_dim)``.
    xdot_train: Their derivatives, same shape and scaling.
    x_validation: Standardized validation delay vectors.
    xdot_validation: Their derivatives.
    config: Hyperparameters.
    verbose: Whether to print per-epoch progress.

  Returns:
    The recorded :class:`TrainingHistory`.
  """
  generator = torch.Generator().manual_seed(config.seed)
  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
  x_train_t = torch.as_tensor(x_train, dtype=torch.float32)
  xdot_train_t = torch.as_tensor(xdot_train, dtype=torch.float32)
  x_validation_t = torch.as_tensor(x_validation, dtype=torch.float32)
  xdot_validation_t = torch.as_tensor(xdot_validation, dtype=torch.float32)

  history = TrainingHistory()
  best_loss = float("inf")
  best_state = None
  epochs_without_improvement = 0
  n_samples = x_train_t.shape[0]

  for epoch in range(config.max_epochs):
    model.train()
    order = torch.randperm(n_samples, generator=generator)
    running = 0.0
    n_batches = 0
    for start in range(0, n_samples, config.batch_size):
      index = order[start : start + config.batch_size]
      optimizer.zero_grad()
      terms = model.losses(x_train_t[index], xdot_train_t[index])
      terms["total"].backward()
      optimizer.step()
      running += float(terms["total"].item())
      n_batches += 1

    model.eval()
    with torch.no_grad():
      validation = float(
        model.losses(x_validation_t, xdot_validation_t)["total"].item()
      )
    history.train_total.append(running / max(n_batches, 1))
    history.validation_total.append(validation)
    history.active_terms.append(int(model.mask.sum().item()))

    if (epoch + 1) % config.threshold_frequency == 0:
      surviving = model.refine_mask()
      if verbose:
        print(f"    epoch {epoch + 1}: refined support to {surviving} terms",
              flush=True)

    if validation < best_loss - 1e-9:
      best_loss = validation
      best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1
      if epochs_without_improvement >= config.patience:
        if verbose:
          print(f"    early stop at epoch {epoch + 1}", flush=True)
        break

    if verbose and (epoch % 10 == 0 or epoch == config.max_epochs - 1):
      print(f"    epoch {epoch:>4}  train {history.train_total[-1]:.5f}  "
            f"val {validation:.5f}  terms {history.active_terms[-1]}", flush=True)

  if best_state is not None:
    model.load_state_dict(best_state)
  return history
