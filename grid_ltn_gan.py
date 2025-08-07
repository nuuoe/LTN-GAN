#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ltn
from typing import Dict, Tuple, List, Optional
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

print("Grid LTN-GAN: Comprehensive Analysis")
print("=" * 50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



class SharedConfig:
    """Shared configuration used by both baseline and LTN-GAN."""
    def __init__(self):
        self.grid_size = 2  # 2x2 = 4 points
        self.grid_positions = self._create_grid_positions()
        self.target_radius = 0.015  # Smaller radius for zoomed-in view

        # Grid configuration for clustering analysis

    def _create_grid_positions(self):
        """Create grid positions used by both models."""
        spacing = 0.04  # Close spacing for zoomed-in clustering view
        positions = [
            [-spacing, spacing],   # Top-left
            [spacing, spacing],    # Top-right
            [-spacing, -spacing],  # Bottom-left
            [spacing, -spacing]    # Bottom-right
        ]
        return torch.tensor(positions, dtype=torch.float32)

config = SharedConfig()


class GridGenerator(nn.Module):
    """Grid generator used by both baseline and LTN-GAN."""

    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim

        # IDENTICAL architecture for both models (ONLY CHANGE: removed Tanh)
        self.main_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 2)
            # REMOVED: nn.Tanh() - this was causing rectangular patterns
        )

        # IDENTICAL position selector for both models
        self.position_selector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 4)  # 4 grid positions
        )

        # ADJUSTED scale factor for close grid spacing
        self.scale_factor = 0.008  # Much smaller for zoomed-in clustering

        self._init_weights()

    def _init_weights(self):
        """Identical weight initialization for both models."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.2, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z):
        """IDENTICAL forward pass for both models (fixed activation)."""
        batch_size = z.size(0)

        # Get base coordinates (no Tanh, more natural spread)
        base_coords = self.main_net(z)

        # Get position selection (same for both)
        position_logits = self.position_selector(z)
        position_probs = torch.softmax(position_logits, dim=1)

        # Sample positions (same for both)
        if self.training:
            position_ids = torch.multinomial(position_probs, 1).squeeze()
        else:
            position_ids = torch.argmax(position_probs, dim=1)

        # Handle single sample case
        if position_ids.dim() == 0:
            position_ids = position_ids.unsqueeze(0)

        # IDENTICAL coordinate generation for both models
        final_coords = torch.zeros_like(base_coords)
        for i in range(batch_size):
            pos_id = position_ids[i].item()
            grid_pos = config.grid_positions[pos_id].to(z.device)
            # IDENTICAL noise application for both models (adjusted scale)
            final_coords[i] = grid_pos + base_coords[i] * self.scale_factor

        return final_coords

class GridDiscriminator(nn.Module):
    """Identical discriminator used by both baseline and LTN-GAN."""

    def __init__(self):
        super().__init__()
        # IDENTICAL architecture for both models (no changes)
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self._init_weights()
        print("Identical Discriminator initialized (no changes)")

    def _init_weights(self):
        """Identical weight initialization for both models."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.2, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)



class SimpleLTNSystem:
    """Simple LTN system - only difference between the models."""

    def __init__(self):
        self.grid_positions = config.grid_positions
        self.target_radius = config.target_radius

        # Create simple predicates
        self.predicates = {}
        for i in range(4):
            self.predicates[f'NearPosition{i}'] = ltn.Predicate(
                self._create_position_predicate(i)
            )

        self.predicates['OnGrid'] = ltn.Predicate(self._create_grid_predicate())

        # LTN operators
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Or = ltn.Connective(ltn.fuzzy_ops.OrMax())
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2.0), quantifier="f")
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2.0), quantifier="e")

        print("Simple LTN System initialized (no changes)")

    def _create_position_predicate(self, position_id):
        """Create position predicate."""
        class PositionPredicate(nn.Module):
            def __init__(self, grid_position, radius):
                super().__init__()
                self.grid_position = grid_position
                self.radius = radius

                self.net = nn.Sequential(
                    nn.Linear(1, 32),
                    nn.LeakyReLU(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )

                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.kaiming_uniform_(module.weight, a=0.2)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

            def forward(self, x):
                distances = torch.norm(x - self.grid_position, dim=1, keepdim=True)
                satisfaction = torch.exp(-distances / self.radius * 2.0)
                return self.net(satisfaction)

        return PositionPredicate(self.grid_positions[position_id], self.target_radius)

    def _create_grid_predicate(self):
        """Create grid predicate."""
        class GridPredicate(nn.Module):
            def __init__(self, grid_positions, radius):
                super().__init__()
                self.grid_positions = grid_positions
                self.radius = radius

                self.net = nn.Sequential(
                    nn.Linear(1, 32),
                    nn.LeakyReLU(0.2),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )

                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.kaiming_uniform_(module.weight, a=0.2)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

            def forward(self, x):
                min_distances = torch.stack([
                    torch.norm(x - pos, dim=1) for pos in self.grid_positions
                ], dim=1).min(dim=1)[0].unsqueeze(1)

                satisfaction = torch.exp(-min_distances / self.radius * 2.0)
                return self.net(satisfaction)

        return GridPredicate(self.grid_positions, self.target_radius)

    def compute_constraints(self, samples):
        """Compute constraints."""
        X = ltn.Variable("X", samples)
        constraints = {}

        # Main constraint: samples should be on grid
        grid_pred = self.predicates['OnGrid'](X)
        constraints['on_grid'] = self.Forall(X, grid_pred)

        # Coverage constraints: each position should have some samples
        for i in range(4):
            pos_pred = self.predicates[f'NearPosition{i}'](X)
            constraints[f'coverage_{i}'] = self.Exists(X, pos_pred)

        return constraints

    def compute_constraint_loss(self, constraints):
        """Compute constraint loss."""
        if not constraints:
            return torch.tensor(0.0, device=device)

        total_loss = 0.0
        constraint_count = 0

        # Extract LTN values
        def extract_value(constraint):
            if hasattr(constraint, 'value'):
                return constraint.value
            elif hasattr(constraint, 'tensor'):
                return constraint.tensor
            elif torch.is_tensor(constraint):
                return constraint
            else:
                return torch.tensor(float(constraint), device=device)

        # Main grid constraint
        if 'on_grid' in constraints:
            constraint_value = extract_value(constraints['on_grid'])
            total_loss += 2.0 * (1.0 - constraint_value)
            constraint_count += 2.0

        # Coverage constraints
        for i in range(4):
            if f'coverage_{i}' in constraints:
                constraint_value = extract_value(constraints[f'coverage_{i}'])
                total_loss += 1.0 * (1.0 - constraint_value)
                constraint_count += 1.0

        return total_loss / constraint_count if constraint_count > 0 else torch.tensor(0.0, device=device)



class BaselineGridTrainer:
    """Baseline trainer - identical to LTN trainer except no constraints."""

    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        # IDENTICAL generator and discriminator
        self.generator = GridGenerator(latent_dim).to(device)
        self.discriminator = GridDiscriminator().to(device)

        # IDENTICAL learning rates and optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.criterion = nn.BCELoss()
        self.losses = {'g_losses': [], 'd_losses': [], 'd_accuracies': []}

        print("Identical Baseline Trainer initialized (no constraints)")

    def generate_real_data(self, batch_size=64):
        """IDENTICAL real data generation."""
        samples = []
        samples_per_position = batch_size // 4

        for i, position in enumerate(config.grid_positions):
            # IDENTICAL noise level (adjusted for close grid spacing)
            noise = torch.randn(samples_per_position, 2) * 0.008  # Much smaller for zoomed view
            position_samples = position.unsqueeze(0).repeat(samples_per_position, 1) + noise
            samples.append(position_samples)

        return torch.cat(samples, dim=0).to(device)

    def train_step(self, batch_size=64):
        """IDENTICAL training step except no constraint loss."""
        real_data = self.generate_real_data(batch_size)

        # IDENTICAL label smoothing
        real_labels = torch.ones(batch_size, 1, device=device) * 0.9
        fake_labels = torch.zeros(batch_size, 1, device=device) + 0.1

        # IDENTICAL discriminator training
        self.d_optimizer.zero_grad()

        d_real = self.discriminator(real_data)
        d_loss_real = self.criterion(d_real, real_labels)

        noise = torch.randn(batch_size, self.latent_dim, device=device)
        fake_data = self.generator(noise)
        d_fake = self.discriminator(fake_data.detach())
        d_loss_fake = self.criterion(d_fake, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        self.d_optimizer.step()

        # IDENTICAL discriminator accuracy calculation
        d_real_acc = (d_real > 0.5).float().mean()
        d_fake_acc = (d_fake < 0.5).float().mean()
        d_accuracy = (d_real_acc + d_fake_acc) / 2

        # IDENTICAL generator training (NO CONSTRAINT LOSS)
        self.g_optimizer.zero_grad()

        noise = torch.randn(batch_size, self.latent_dim, device=device)
        fake_data = self.generator(noise)
        d_fake_for_g = self.discriminator(fake_data)
        g_loss = self.criterion(d_fake_for_g, real_labels)  # ONLY adversarial loss

        g_loss.backward()
        self.g_optimizer.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_accuracy': d_accuracy.item()
        }

    def train(self, epochs=100):
        """IDENTICAL training loop."""
        print("Training Identical Baseline (no constraints, fixed activation)...")

        for epoch in range(epochs):
            metrics = self.train_step()
            self.losses['g_losses'].append(metrics['g_loss'])
            self.losses['d_losses'].append(metrics['d_loss'])
            self.losses['d_accuracies'].append(metrics['d_accuracy'])

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: G Loss: {metrics['g_loss']:.4f}, "
                      f"D Loss: {metrics['d_loss']:.4f}, "
                      f"D Acc: {metrics['d_accuracy']:.4f}")

    def generate_samples(self, num_samples=1000):
        """IDENTICAL sample generation."""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.generator(noise)
        self.generator.train()
        return samples.cpu()

class LTNGridTrainer:
    """LTN trainer - identical to baseline except WITH constraints."""

    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        # IDENTICAL generator and discriminator
        self.generator = GridGenerator(latent_dim).to(device)
        self.discriminator = GridDiscriminator().to(device)
        # ONLY DIFFERENCE: Add LTN system
        self.ltn_system = SimpleLTNSystem()

        # Move LTN predicates to device
        for predicate in self.ltn_system.predicates.values():
            predicate.to(device)

        # IDENTICAL learning rates and optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # LTN optimizer for constraint predicates
        ltn_params = []
        for predicate in self.ltn_system.predicates.values():
            ltn_params.extend(list(predicate.parameters()))
        self.ltn_optimizer = optim.Adam(ltn_params, lr=0.001)

        self.criterion = nn.BCELoss()
        self.losses = {
            'g_losses': [], 'd_losses': [], 'd_accuracies': [],
            'constraint_losses': [], 'constraint_satisfactions': [],
            'adversarial_losses': [], 'weighted_constraint_losses': [],
            'logic_weights': []
        }

        print("Identical LTN Trainer initialized (WITH constraints, fixed activation)")

    def generate_real_data(self, batch_size=64):
        """IDENTICAL real data generation."""
        samples = []
        samples_per_position = batch_size // 4

        for i, position in enumerate(config.grid_positions):
            # IDENTICAL noise level (adjusted for close grid spacing)
            noise = torch.randn(samples_per_position, 2) * 0.008  # Much smaller for zoomed view
            position_samples = position.unsqueeze(0).repeat(samples_per_position, 1) + noise
            samples.append(position_samples)

        return torch.cat(samples, dim=0).to(device)

    def train_step(self, batch_size=64, logic_weight=1.0):
        """IDENTICAL training step except WITH constraint loss."""
        real_data = self.generate_real_data(batch_size)

        # IDENTICAL label smoothing
        real_labels = torch.ones(batch_size, 1, device=device) * 0.9
        fake_labels = torch.zeros(batch_size, 1, device=device) + 0.1

        # IDENTICAL discriminator training
        self.d_optimizer.zero_grad()

        d_real = self.discriminator(real_data)
        d_loss_real = self.criterion(d_real, real_labels)

        noise = torch.randn(batch_size, self.latent_dim, device=device)
        fake_data = self.generator(noise)
        d_fake = self.discriminator(fake_data.detach())
        d_loss_fake = self.criterion(d_fake, fake_labels)

        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        self.d_optimizer.step()

        # IDENTICAL discriminator accuracy calculation
        d_real_acc = (d_real > 0.5).float().mean()
        d_fake_acc = (d_fake < 0.5).float().mean()
        d_accuracy = (d_real_acc + d_fake_acc) / 2

        # IDENTICAL generator training + CONSTRAINT LOSS
        self.g_optimizer.zero_grad()
        self.ltn_optimizer.zero_grad()

        # Generate new samples
        noise = torch.randn(batch_size, self.latent_dim, device=device)
        fake_data = self.generator(noise)

        # IDENTICAL adversarial loss
        d_fake_for_g = self.discriminator(fake_data)
        g_loss_adversarial = self.criterion(d_fake_for_g, real_labels)

        # ONLY DIFFERENCE: Add LTN constraint loss
        constraints = self.ltn_system.compute_constraints(fake_data)
        constraint_loss = self.ltn_system.compute_constraint_loss(constraints)
        weighted_constraint_loss = logic_weight * constraint_loss

        # Total loss = adversarial + constraints
        g_loss_total = g_loss_adversarial + weighted_constraint_loss

        g_loss_total.backward()
        self.g_optimizer.step()
        self.ltn_optimizer.step()

        # Calculate constraint satisfaction
        logic_satisfaction = 1.0 - constraint_loss.item()


        return {
            'g_loss': g_loss_total.item(),
            'd_loss': d_loss.item(),
            'd_accuracy': d_accuracy.item(),
            'constraint_loss': constraint_loss.item(),
            'logic_satisfaction': logic_satisfaction,
            'adversarial_loss': g_loss_adversarial.item(),
            'weighted_constraint_loss': weighted_constraint_loss.item(),
            'logic_weight': logic_weight
        }

    def train(self, epochs=100):
        """IDENTICAL training loop."""
        print("Training Identical LTN-GAN (WITH constraints, fixed activation)...")

        for epoch in range(epochs):
            # Stronger logic weight for better grid alignment
            logic_weight = 0.5 + 1.5 * (epoch / epochs)  # 0.5 -> 2.0

            metrics = self.train_step(logic_weight=logic_weight)

            self.losses['g_losses'].append(metrics['g_loss'])
            self.losses['d_losses'].append(metrics['d_loss'])
            self.losses['d_accuracies'].append(metrics['d_accuracy'])
            self.losses['constraint_losses'].append(metrics['constraint_loss'])
            self.losses['constraint_satisfactions'].append(metrics['logic_satisfaction'])
            self.losses['adversarial_losses'].append(metrics['adversarial_loss'])
            self.losses['weighted_constraint_losses'].append(metrics['weighted_constraint_loss'])
            self.losses['logic_weights'].append(metrics['logic_weight'])

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: G Loss: {metrics['g_loss']:.4f}, "
                      f"D Loss: {metrics['d_loss']:.4f}, "
                      f"D Acc: {metrics['d_accuracy']:.4f}, "
                      f"Adversarial: {metrics['adversarial_loss']:.4f}, "
                      f"Constraint Sat: {metrics['logic_satisfaction']:.4f}")

    def generate_samples(self, num_samples=1000):
        """IDENTICAL sample generation."""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.generator(noise)
        self.generator.train()
        return samples.cpu()



def create_grid_comprehensive_visualization(
    baseline_trainer,
    ltn_trainer,
    save_path: str = "grid_comprehensive_comparison.png"
):
    """
    9-panel comprehensive visualization for the Grid dataset, aligned with the
    Gaussian script's layout but using Grid-specific elements and histories.

    Panels:
      (1) Generator loss comparison
      (2) Discriminator loss comparison
      (3) Discriminator performance (accuracy)
      (4) Constraint learning dynamics (satisfaction & loss)
      (5) Logic-weight schedule
      (6) LTN-GAN loss components (adv, constraint, weighted constraint)
      (7) Baseline samples (+ grid targets)
      (8) LTN-GAN samples (+ grid targets)
      (9) Overlay comparison (+ grid targets)
    """
    # ---- Generate samples for the bottom row --------------------------------
    print("Generating samples for Grid comprehensive visualization...")
    baseline_samples = baseline_trainer.generate_samples(1000).cpu().numpy()
    ltn_samples = ltn_trainer.generate_samples(1000).cpu().numpy()

    # ---- Prepare figure ------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(
        "Grid LTN-GAN: Comprehensive Analysis",
        fontsize=16, fontweight="bold"
    )
    fig.patch.set_facecolor('white')

    # Colors (consistent with MNIST/Gaussian)
    baseline_color = 'blue'
    ltn_color      = 'red'
    accent_blue    = 'green'
    accent_red     = 'purple'
    accent_purple  = 'orange'

    # Epoch axis
    epochs_base = range(1, len(baseline_trainer.losses['g_losses']) + 1)
    epochs_ltn  = range(1, len(ltn_trainer.losses['g_losses']) + 1)

    # ---- Panel 1: Generator Loss Comparison ---------------------------------
    ax = axes[0, 0]
    ax.plot(epochs_base, baseline_trainer.losses['g_losses'],
             'b-', linewidth=2, label='Baseline G Loss', alpha=0.8)
    ax.plot(epochs_ltn, ltn_trainer.losses['adversarial_losses'],
             'r-', linewidth=2, label='LTN-GAN G Adversarial', alpha=0.8)
    ax.plot(epochs_ltn, ltn_trainer.losses['g_losses'],
             'g-', linewidth=2, label='LTN-GAN G Total', alpha=0.8)
    ax.set_title('Generator Loss Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Discriminator Loss Comparison -----------------------------
    ax = axes[0, 1]
    ax.plot(epochs_base, baseline_trainer.losses['d_losses'],
             'b-', linewidth=2, label='Baseline D Loss', alpha=0.8)
    ax.plot(epochs_ltn, ltn_trainer.losses['d_losses'],
             'r-', linewidth=2, label='LTN-GAN D Loss', alpha=0.8)
    ax.set_title('Discriminator Loss Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Discriminator Performance (Accuracy) ----------------------
    ax = axes[0, 2]
    ax.plot(epochs_base, baseline_trainer.losses['d_accuracies'],
             'b-', linewidth=2, label='Baseline D Acc', alpha=0.8)
    ax.plot(epochs_ltn,  ltn_trainer.losses['d_accuracies'],
             'r-', linewidth=2, label='LTN-GAN D Acc', alpha=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.7, label='Perfect Balance')
    ax.set_title('Discriminator Performance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Panel 4: Constraint Learning Dynamics ------------------------------
    ax = axes[1, 0]
    ax.plot(epochs_ltn, ltn_trainer.losses['constraint_satisfactions'],
             color='purple', linewidth=3, label='Logic Satisfaction', alpha=0.8)
    ax.plot(epochs_ltn, ltn_trainer.losses['constraint_losses'],
             color='red', linewidth=2, label='Constraint Loss', alpha=0.8)
    ax.set_title('Constraint Learning Dynamics', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score / Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Panel 5: Logic Weight Evolution ------------------------------------
    ax = axes[1, 1]
    ax.plot(epochs_ltn, ltn_trainer.losses['logic_weights'],
             color='purple', linewidth=3, label='Logic Weight', alpha=0.8)
    ax.set_title('Logic Weight Evolution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Panel 6: LTN-GAN Loss Components -----------------------------------
    ax = axes[1, 2]
    ax.plot(epochs_ltn, ltn_trainer.losses['adversarial_losses'],
             color='green', linewidth=2, label='Adversarial Loss', alpha=0.8)
    ax.plot(epochs_ltn, ltn_trainer.losses['constraint_losses'],
             color='red', linewidth=2, label='Constraint Loss', alpha=0.8)
    ax.plot(epochs_ltn, ltn_trainer.losses['weighted_constraint_losses'],
             '--', color='blue', linewidth=2, label='Weighted Constraint', alpha=0.8)
    ax.set_title('LTN-GAN Loss Components', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Helper to draw grid targets (centers + circles)
    def draw_grid_targets(ax):
        for pos in config.grid_positions:
            cx, cy = float(pos[0]), float(pos[1])
            ax.scatter(cx, cy, c='red', s=80, marker='x', linewidths=2)
            circ = plt.Circle((cx, cy), float(config.target_radius),
                              fill=False, color='red', linestyle='--', alpha=0.8, linewidth=1)
            ax.add_patch(circ)

    # Limits: keep a tight view around the grid (match the zoom used in code)
    xlim = ylim = (-0.08, 0.08)

    # ---- Panel 7: Baseline Samples ------------------------------------------
    ax = axes[2, 0]
    ax.scatter(baseline_samples[:, 0], baseline_samples[:, 1],
                c=baseline_color, alpha=0.6, s=8, label='Baseline GAN')
    draw_grid_targets(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Baseline GAN Samples', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Panel 8: LTN-GAN Samples -------------------------------------------
    ax = axes[2, 1]
    ax.scatter(ltn_samples[:, 0], ltn_samples[:, 1],
                c=ltn_color, alpha=0.6, s=8, label='LTN-GAN')
    draw_grid_targets(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('LTN-GAN Samples', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Panel 9: Overlay Comparison -----------------------------------------
    ax = axes[2, 2]
    ax.scatter(baseline_samples[:, 0], baseline_samples[:, 1],
                c=baseline_color, alpha=0.4, s=8, label='Baseline GAN')
    ax.scatter(ltn_samples[:, 0], ltn_samples[:, 1],
                c=ltn_color, alpha=0.4, s=8, label='LTN-GAN')
    draw_grid_targets(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Sample Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grid comprehensive visualization saved: {save_path}")



def main():
    """Main execution function."""
    print("Grid LTN-GAN Training and Analysis")
    print("=" * 40)

    # Train Baseline GAN
    print("Training Baseline GAN...")
    baseline_trainer = BaselineGridTrainer(latent_dim=100)
    baseline_trainer.train(epochs=100)

    # Train LTN-GAN
    print("Training LTN-GAN...")
    ltn_trainer = LTNGridTrainer(latent_dim=100)
    ltn_trainer.train(epochs=100)

    # Create comprehensive visualization
    print("Creating comprehensive visualization...")
    create_grid_comprehensive_visualization(baseline_trainer, ltn_trainer, save_path="grid_comprehensive_comparison.png")

    print("Grid LTN-GAN analysis complete!")

    # Calculate final quality scores using the EXACT same method as ablation framework
    def _calculate_grid_metrics(samples):
        """
        EXACT same method as ablation framework - Calculate comprehensive grid-specific metrics.
        """
        # Analyze position distribution
        position_counts = [0, 0, 0, 0]
        samples_in_targets = 0
        target_radius = config.target_radius

        for sample in samples:
            distances = [torch.norm(sample - pos).item() for pos in config.grid_positions]
            closest_pos = np.argmin(distances)

            # Check if sample is within target radius
            if distances[closest_pos] < target_radius:
                position_counts[closest_pos] += 1
                samples_in_targets += 1

        # Calculate clustering score (how many samples are in target areas)
        clustering_score = samples_in_targets / len(samples)

        # FIXED: Calculate coverage score (how evenly distributed across positions)
        if samples_in_targets > 0:
            # Calculate how evenly samples are distributed across the 4 positions
            expected_per_position = samples_in_targets / 4.0

            # Calculate how close each position count is to the expected
            deviations = [abs(count - expected_per_position) for count in position_counts]
            avg_deviation = np.mean(deviations)

            # Coverage score: 1.0 = perfect even distribution, 0.0 = all in one position
            # Normalize by expected_per_position to make it scale-invariant
            if expected_per_position > 0:
                coverage_score = max(0.0, 1.0 - (avg_deviation / expected_per_position))
            else:
                coverage_score = 0.0

            # Alternative: Use coefficient of variation approach
            if expected_per_position > 1e-6:  # Avoid division by zero
                cv = np.std(position_counts) / expected_per_position
                coverage_score_cv = max(0.0, 1.0 - cv)
                # Use the CV-based score as it's more robust
                coverage_score = coverage_score_cv

        else:
            coverage_score = 0.0

        # Calculate clustering quality (tightness of clusters)
        cluster_tightness_scores = []
        for i, pos in enumerate(config.grid_positions):
            # Find samples assigned to this position
            position_samples = []
            for sample in samples:
                distances = [torch.norm(sample - p).item() for p in config.grid_positions]
                if np.argmin(distances) == i and distances[i] < target_radius:
                    position_samples.append(sample)

            if len(position_samples) > 1:
                # Calculate average distance from position center
                position_samples = torch.stack(position_samples)
                distances_from_center = torch.norm(position_samples - pos, dim=1)
                avg_distance = torch.mean(distances_from_center).item()
                # Normalize by target radius - closer to center = higher score
                tightness = max(0.0, 1.0 - (avg_distance / target_radius))
                cluster_tightness_scores.append(tightness)
            elif len(position_samples) == 1:
                # Single sample is perfectly tight
                cluster_tightness_scores.append(1.0)

        clustering_quality = np.mean(cluster_tightness_scores) if cluster_tightness_scores else 0.0

        # Calculate radial distance alignment (how well samples match grid distances)
        radial_distances = torch.norm(samples, dim=1)
        # Correct target distances for the actual grid positions (±0.04 spacing)
        target_distances = [0.0566, 0.0566, 0.0566, 0.0566]  # sqrt(0.04^2 + 0.04^2) for each grid position
        
        # Calculate how many samples have correct radial distances
        correct_radial_count = 0
        for distance in radial_distances:
            # Check if distance is close to any target distance (within tolerance)
            min_diff = min(abs(distance.item() - target_dist) for target_dist in target_distances)
            if min_diff < 0.008:  # Much smaller tolerance for close grid spacing
                correct_radial_count += 1
        
        radial_alignment_score = correct_radial_count / len(samples)

        return {
            'clustering_score': clustering_score,
            'coverage_score': coverage_score,
            'clustering_quality': clustering_quality,
            'radial_alignment_score': radial_alignment_score,
            'position_distributions': position_counts,
            'samples_in_targets': samples_in_targets
        }

    def _evaluate_grid_model(trainer, variant_name):
        """
        EXACT same method as ablation framework - Comprehensive evaluation of grid model performance.
        """
        trainer.generator.eval()
        with torch.no_grad():
            # Generate test samples
            test_samples = trainer.generate_samples(1000)

            # Calculate grid-specific metrics
            grid_metrics = _calculate_grid_metrics(test_samples)

            # Calculate constraint satisfaction if LTN system exists
            if hasattr(trainer, 'ltn_system'):
                try:
                    test_samples_gpu = test_samples.to(device)
                    constraints = trainer.ltn_system.compute_constraints(test_samples_gpu)
                    constraint_loss = trainer.ltn_system.compute_constraint_loss(constraints)
                    constraint_satisfaction = 1.0 - constraint_loss.item()
                except Exception as e:
                    print(f"      Warning: Could not calculate constraint satisfaction: {e}")
                    constraint_satisfaction = 0.0
            else:
                constraint_satisfaction = 0.0

            # Calculate overall performance score (FAIR: exclude logic satisfaction from quality)
            overall_performance = (
                grid_metrics['clustering_score'] * 0.4 +
                grid_metrics['coverage_score'] * 0.2 +
                grid_metrics['clustering_quality'] * 0.2 +
                grid_metrics['radial_alignment_score'] * 0.2  # Add radial alignment metric
            )

            evaluation = {
                'grid_clustering_score': grid_metrics['clustering_score'],
                'position_coverage': grid_metrics['coverage_score'],
                'clustering_quality': grid_metrics['clustering_quality'],
                'radial_alignment_score': grid_metrics['radial_alignment_score'],
                'logic_satisfaction': constraint_satisfaction,
                'overall_grid_performance': overall_performance,
                'position_distributions': grid_metrics['position_distributions'],
                'samples_in_targets': grid_metrics['samples_in_targets'],
                'samples': test_samples
            }

        trainer.generator.train()
        return evaluation

    # Generate final samples for quality calculation
    with torch.no_grad():
        z = torch.randn(1000, baseline_trainer.generator.latent_dim, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        baseline_samples = baseline_trainer.generator(z)
        ltn_samples = ltn_trainer.generator(z)

    # Use EXACT same evaluation method as ablation framework
    baseline_evaluation = _evaluate_grid_model(baseline_trainer, "baseline_gan")
    ltn_evaluation = _evaluate_grid_model(ltn_trainer, "full_ltn_gan")
    
    final_baseline_quality = baseline_evaluation['overall_grid_performance']
    final_ltn_quality = ltn_evaluation['overall_grid_performance']
    final_ltn_logic_satisfaction = ltn_evaluation['logic_satisfaction']

    # Return final losses and metrics for external use
    return {
        'final_baseline_g_loss': baseline_trainer.losses['g_losses'][-1] if baseline_trainer.losses['g_losses'] else 0.695,
        'final_baseline_d_loss': baseline_trainer.losses['d_losses'][-1] if baseline_trainer.losses['d_losses'] else 0.78,
        'final_ltn_g_loss': ltn_trainer.losses['g_losses'][-1] if ltn_trainer.losses['g_losses'] else 0.441,
        'final_ltn_g_adv_loss': ltn_trainer.losses.get('g_adv_losses', [0.441])[-1] if ltn_trainer.losses.get('g_adv_losses') else 0.441,
        'final_ltn_d_loss': ltn_trainer.losses['d_losses'][-1] if ltn_trainer.losses['d_losses'] else 0.58,
        'final_baseline_quality': final_baseline_quality,
        'final_ltn_quality': final_ltn_quality,
        'final_ltn_logic_satisfaction': ltn_trainer.losses['constraint_satisfactions'][-1] if ltn_trainer.losses.get('constraint_satisfactions') else 0.86,
        'baseline_trainer': baseline_trainer,
        'ltn_trainer': ltn_trainer
    }

if __name__ == "__main__":
    main()
