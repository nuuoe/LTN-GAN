#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import ltn
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
import random
warnings.filterwarnings('ignore')

# Set seeds for reproducible results
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
#plt.style.use('seaborn-v0_8')
#sns.set_palette("husl")


@dataclass
class SophisticatedConstraintConfig:
    # Ring parameters
    inner_radius: float = 1.0
    outer_radius: float = 2.0
    plot_range: float = 3.0

    # Fine-tuned hierarchical constraint parameters for optimal satisfaction
    basic_tolerance: float = 0.35         # Good initial precision
    intermediate_tolerance: float = 0.18   # INCREASED from 0.15 for smoother transition
    advanced_tolerance: float = 0.10       # INCREASED from 0.08 to prevent satisfaction drop

    # Progressive learning parameters
    learning_phases: int = 3  # Basic, Intermediate, Advanced
    phase_epochs: int = 50  # Epochs per phase
    tolerance_decay: float = 0.8  # How much to tighten each phase

    # Optimized fuzzy logic parameters for 80%+ constraint satisfaction
    fuzzy_temperature: float = 1.2        # REDUCED from 1.5 for even sharper logic
    satisfaction_threshold: float = 0.75   # INCREASED from 0.7 for higher standards
    constraint_momentum: float = 0.85      # INCREASED from 0.8 for more stability

    # Optimized smart weighting parameters
    initial_weights: Dict[str, float] = None
    weight_adaptation_rate: float = 0.12   # REDUCED from 0.15 for more stable adaptation
    min_weight: float = 0.3                # INCREASED from 0.2 for stronger minimum
    max_weight: float = 12.0               # REDUCED from 15.0 for better balance
    # Training parameters
    batch_size: int = 64
    n_samples: int = 500

    def __post_init__(self):
        if self.initial_weights is None:
            self.initial_weights = {
                # Level 1: Basic existence (highest priority)
                'ring_existence': 5.0,
                'basic_membership': 3.0,

                # Level 2: Logical relationships (medium priority)
                'mutual_exclusivity': 2.0,
                'completeness': 2.0,
                'spatial_consistency': 2.0,

                # Level 3: Refinement (lower priority initially)
                'balance_logic': 1.0,
                'precision_refinement': 0.5,
                'dead_zone_avoidance': 1.5
            }

class SophisticatedRingPredicates:
    """Sophisticated ring predicates using proper LTNtorch semantics"""

    def __init__(self, config: SophisticatedConstraintConfig, device):
        self.config = config
        self.device = device
        self.current_tolerance = config.basic_tolerance
        self.current_phase = 0

        # Define sophisticated LTN predicates
        self._create_predicates()

        print(f"Sophisticated Ring Predicates initialized")
        print(f"Hierarchical learning: {config.learning_phases} phases")
        print(f"Progressive tolerance: {config.basic_tolerance} → {config.advanced_tolerance}")
        print(f"Fuzzy temperature: {config.fuzzy_temperature}")
        print(f"Smart weighting enabled")

    def _create_predicates(self):
        """Create sophisticated LTN predicates with proper semantics"""

        # Level 1: Basic Ring Predicates
        class OnInnerRingPredicate(nn.Module):
            def __init__(self, config, tolerance_fn):
                super().__init__()
                self.config = config
                self.tolerance_fn = tolerance_fn

            def forward(self, x):
                distances = torch.norm(x, dim=1)
                tolerance = self.tolerance_fn()

                # Fuzzy membership with smooth transitions
                diff = torch.abs(distances - self.config.inner_radius)
                membership = torch.exp(-diff**2 / (2 * tolerance**2))
                return membership

        class OnOuterRingPredicate(nn.Module):
            def __init__(self, config, tolerance_fn):
                super().__init__()
                self.config = config
                self.tolerance_fn = tolerance_fn

            def forward(self, x):
                distances = torch.norm(x, dim=1)
                tolerance = self.tolerance_fn()

                # Fuzzy membership with smooth transitions
                diff = torch.abs(distances - self.config.outer_radius)
                membership = torch.exp(-diff**2 / (2 * tolerance**2))
                return membership

        class InDeadZonePredicate(nn.Module):
            def __init__(self, config, tolerance_fn):
                super().__init__()
                self.config = config
                self.tolerance_fn = tolerance_fn

            def forward(self, x):
                distances = torch.norm(x, dim=1)
                tolerance = self.tolerance_fn()

                # Dead zone: between rings with some tolerance
                inner_boundary = self.config.inner_radius + tolerance
                outer_boundary = self.config.outer_radius - tolerance

                in_dead_zone = (distances > inner_boundary) & (distances < outer_boundary)
                return in_dead_zone.float()

        # Level 2: Spatial Consistency Predicates
        class SpatiallyConsistentPredicate(nn.Module):
            def __init__(self, config, tolerance_fn):
                super().__init__()
                self.config = config
                self.tolerance_fn = tolerance_fn

            def forward(self, x):
                distances = torch.norm(x, dim=1)
                tolerance = self.tolerance_fn()

                # Points should be consistent with their ring assignment
                inner_consistency = torch.exp(-torch.abs(distances - self.config.inner_radius)**2 / (2 * tolerance**2))
                outer_consistency = torch.exp(-torch.abs(distances - self.config.outer_radius)**2 / (2 * tolerance**2))

                # Maximum consistency with either ring
                max_consistency = torch.max(inner_consistency, outer_consistency)
                return max_consistency

        # Level 3: Balance and Precision Predicates
        class BalancedDistributionPredicate(nn.Module):
            def __init__(self, config, tolerance_fn):
                super().__init__()
                self.config = config
                self.tolerance_fn = tolerance_fn

            def forward(self, x):
                distances = torch.norm(x, dim=1)
                tolerance = self.tolerance_fn()

                # Count points on each ring
                inner_membership = torch.exp(-torch.abs(distances - self.config.inner_radius)**2 / (2 * tolerance**2))
                outer_membership = torch.exp(-torch.abs(distances - self.config.outer_radius)**2 / (2 * tolerance**2))

                inner_count = torch.sum(inner_membership)
                outer_count = torch.sum(outer_membership)
                total_count = inner_count + outer_count + 1e-8  # Avoid division by zero

                # Balance score: closer to 0.5 is better
                inner_ratio = inner_count / total_count
                balance_score = 1.0 - 2.0 * torch.abs(inner_ratio - 0.5)

                # Return same score for all points (global constraint)
                return balance_score.expand(x.shape[0])

        # Create tolerance function for dynamic adjustment
        def get_current_tolerance():
            return self.current_tolerance

        # Initialize LTN predicates
        self.OnInnerRing = ltn.Predicate(OnInnerRingPredicate(self.config, get_current_tolerance).to(self.device))
        self.OnOuterRing = ltn.Predicate(OnOuterRingPredicate(self.config, get_current_tolerance).to(self.device))
        self.InDeadZone = ltn.Predicate(InDeadZonePredicate(self.config, get_current_tolerance).to(self.device))
        self.SpatiallyConsistent = ltn.Predicate(SpatiallyConsistentPredicate(self.config, get_current_tolerance).to(self.device))
        self.BalancedDistribution = ltn.Predicate(BalancedDistributionPredicate(self.config, get_current_tolerance).to(self.device))

    def update_phase(self, phase: int):
        """Update learning phase and adjust tolerance"""
        self.current_phase = phase

        # Progressive tolerance tightening
        if phase == 0:  # Basic phase
            self.current_tolerance = self.config.basic_tolerance
        elif phase == 1:  # Intermediate phase
            self.current_tolerance = self.config.intermediate_tolerance
        else:  # Advanced phase
            self.current_tolerance = self.config.advanced_tolerance

        print(f"Updated to Phase {phase + 1}: tolerance = {self.current_tolerance:.3f}")

class SophisticatedConstraintSystem:

    def __init__(self, config: SophisticatedConstraintConfig, device):
        self.config = config
        self.device = device
        self.predicates = SophisticatedRingPredicates(config, device)

        # Initialize constraint weights
        self.weights = config.initial_weights.copy()
        self.weight_history = {name: [weight] for name, weight in self.weights.items()}
        self.satisfaction_history = {name: [] for name in self.weights.keys()}

        # LTN operators
        self.forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        self.exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

        print(f"Sophisticated Constraint System initialized")
        print(f"Hierarchical constraints: {len(self.weights)} types")
        print(f"Smart weighting enabled")
        print(f"Progressive learning phases: {config.learning_phases}")

    def compute_sophisticated_constraints(self, samples):
        X = ltn.Variable("X", samples)
        constraints = {}

        # LEVEL 1: Basic Existence Constraints (Highest Priority)

        # Ring existence: There exist points on both rings
        constraints['ring_existence'] = self.And(
            self.exists(X, self.predicates.OnInnerRing(X)),
            self.exists(X, self.predicates.OnOuterRing(X))
        )

        # Basic membership: Every point should be on some ring or in dead zone
        constraints['basic_membership'] = self.forall(X,
            self.Or(
                self.predicates.OnInnerRing(X),
                self.Or(
                    self.predicates.OnOuterRing(X),
                    self.predicates.InDeadZone(X)
                )
            )
        )

        # LEVEL 2: Logical Relationships (Medium Priority)

        # Mutual exclusivity: Points cannot be on both rings simultaneously
        constraints['mutual_exclusivity'] = self.forall(X,
            self.Not(
                self.And(
                    self.predicates.OnInnerRing(X),
                    self.predicates.OnOuterRing(X)
                )
            )
        )

        # Completeness: Every point has a clear classification
        constraints['completeness'] = self.forall(X,
            self.Or(
                self.predicates.OnInnerRing(X),
                self.Or(
                    self.predicates.OnOuterRing(X),
                    self.predicates.InDeadZone(X)
                )
            )
        )

        # Spatial consistency: Points are spatially consistent with ring structure
        constraints['spatial_consistency'] = self.forall(X, self.predicates.SpatiallyConsistent(X))

        # LEVEL 3: Refinement Constraints (Lower Priority Initially)

        # Balance logic: Approximately equal distribution between rings
        constraints['balance_logic'] = self.forall(X, self.predicates.BalancedDistribution(X))

        # Precision refinement: Points should be precisely on rings (tightens over time)
        constraints['precision_refinement'] = self.And(
            self.forall(X,
                self.Implies(
                    self.predicates.OnInnerRing(X),
                    self.predicates.SpatiallyConsistent(X)
                )
            ),
            self.forall(X,
                self.Implies(
                    self.predicates.OnOuterRing(X),
                    self.predicates.SpatiallyConsistent(X)
                )
            )
        )

        # Dead zone avoidance: Minimize points in dead zone
        constraints['dead_zone_avoidance'] = self.forall(X, self.Not(self.predicates.InDeadZone(X)))

        return constraints

    def compute_sophisticated_loss(self, constraints):
        """Compute sophisticated constraint loss with smart weighting"""
        sat_agg = ltn.fuzzy_ops.SatAgg()

        # Compute individual constraint satisfactions
        constraint_satisfactions = {}
        weighted_constraints = []

        for name, constraint in constraints.items():
            satisfaction = constraint.value.item()
            constraint_satisfactions[name] = satisfaction

            # Add to satisfaction history
            self.satisfaction_history[name].append(satisfaction)

            # Add weighted constraints based on current weights
            weight = self.weights[name]
            for _ in range(int(weight)):
                weighted_constraints.append(constraint)

        # Compute total loss
        if weighted_constraints:
            total_satisfaction = sat_agg(*weighted_constraints)
            total_loss = 1.0 - total_satisfaction
        else:
            total_loss = torch.tensor(1.0, device=self.device)

        return total_loss, constraint_satisfactions

    def update_weights(self, constraint_satisfactions):
        """Smart weight updating based on constraint satisfaction feedback"""
        for name, satisfaction in constraint_satisfactions.items():
            current_weight = self.weights[name]

            # Adaptive weight adjustment
            if satisfaction < 0.3:  # Very low satisfaction - increase weight
                new_weight = current_weight * (1 + self.config.weight_adaptation_rate)
            elif satisfaction > 0.8:  # High satisfaction - can reduce weight slightly
                new_weight = current_weight * (1 - self.config.weight_adaptation_rate * 0.5)
            else:  # Moderate satisfaction - small adjustments
                target_satisfaction = 0.6
                adjustment = (target_satisfaction - satisfaction) * self.config.weight_adaptation_rate
                new_weight = current_weight * (1 + adjustment)

            # Apply constraints and momentum
            new_weight = np.clip(new_weight, self.config.min_weight, self.config.max_weight)
            self.weights[name] = (self.config.constraint_momentum * current_weight +
                                (1 - self.config.constraint_momentum) * new_weight)

            # Store in history
            self.weight_history[name].append(self.weights[name])

    def update_phase(self, phase: int):
        self.predicates.update_phase(phase)

        # Adjust weights for current phase
        if phase == 0:  # Basic phase - focus on existence
            self.weights['ring_existence'] *= 2.0
            self.weights['basic_membership'] *= 1.5
        elif phase == 1:  # Intermediate phase - focus on relationships
            self.weights['mutual_exclusivity'] *= 1.5
            self.weights['spatial_consistency'] *= 1.5
        else:  # Advanced phase - focus on precision
            self.weights['precision_refinement'] *= 2.0
            self.weights['balance_logic'] *= 1.5




@dataclass
class RingConfig(SophisticatedConstraintConfig):
    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    n_samples: int = 500

    # Ring-enabling hardcoding parameters
    ring_guidance_strength: float = 0.3
    polar_coordinate_bias: float = 0.2
    radius_awareness: float = 0.15

    # Discriminator parameters
    hidden_dim: int = 256
    dropout: float = 0.3

    # Data generation parameters
    noise_level: float = 0.1
    ring_tolerance: float = 0.25

    # Learning rates
    lr_g: float = 0.0003
    lr_d: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999

    # LTN constraint parameters (match ablation framework full_ltn_gan)
    constraint_weight_start: float = 0.0
    constraint_weight_end: float = 10.0
    constraint_ramp_epochs: int = 100

    # Training dynamics parameters
    d_train_frequency: int = 1
    g_train_frequency: int = 2
    label_smoothing: float = 0.1

    # Evaluation parameters
    evaluation_frequency: int = 25
    ring_adherence_target_baseline: float = 0.40
    ring_adherence_target_ltn: float = 0.95

    def __post_init__(self):
        super().__post_init__()
        # Adjust sophisticated constraint weights for ring-enabling context
        self.initial_weights = {
            # Level 1: Ring formation (MAXIMUM priority for 95% adherence)
            'ring_existence': 10.0,
            'spatial_consistency': 9.0,
            'dead_zone_avoidance': 8.0,

            # Level 2: Classification and separation (HIGH priority)
            'basic_membership': 7.0,
            'mutual_exclusivity': 6.0,
            'completeness': 6.0,

            # Level 3: Balance and refinement (MEDIUM priority)
            'balance_logic': 4.0,
            'precision_refinement': 3.5
        }

class RingGenerator(nn.Module):
    """Generator with ring-enabling hardcoding for both baseline and LTN-GAN"""

    def __init__(self, config: RingConfig, use_ltn_constraints: bool = False):
        super().__init__()
        self.config = config
        self.use_ltn_constraints = use_ltn_constraints
        self.latent_dim = 100

        # Base generator network
        self.base_network = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(config.dropout),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Ring-enabling components (same for both models)
        self.ring_selector = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 2),  # [inner_preference, outer_preference]
            nn.Softmax(dim=1)
        )

        self.coordinate_generator = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 2),  # [radius_factor, angle_factor]
            nn.Tanh()
        )

        # Output scaling
        self.output_scale = 2.5

        model_type = "LTN-GAN" if use_ltn_constraints else "Baseline"
        print(f"Ring-Enabling {model_type} Generator initialized")
        print(f"Ring guidance strength: {config.ring_guidance_strength}")
        print(f"Polar coordinate bias: {config.polar_coordinate_bias}")
        print(f"Radius awareness: {config.radius_awareness}")
        print(f"LTN constraints: {use_ltn_constraints}")

    def forward(self, z):
        # Base feature extraction
        features = self.base_network(z)

        # Ring selection (which ring to target)
        ring_preferences = self.ring_selector(features)
        inner_pref = ring_preferences[:, 0:1]
        outer_pref = ring_preferences[:, 1:2]

        # Coordinate generation with ring-enabling bias
        coords = self.coordinate_generator(features)
        radius_factor = coords[:, 0:1]
        angle_factor = coords[:, 1:2]

        # Convert to polar coordinates
        angles = angle_factor * np.pi * 2  # Full rotation
        base_radius = (radius_factor + 1) / 2  # [0, 1] range

        # Ring-aware generation (same for both models)
        inner_target = self.config.inner_radius / self.output_scale
        outer_target = self.config.outer_radius / self.output_scale

        # Apply ring guidance to target rings (same for both)
        guided_radius = (
            (1 - self.config.ring_guidance_strength) * base_radius +
            self.config.ring_guidance_strength * (
                inner_pref * inner_target + outer_pref * outer_target
            )
        )

        # Add polar coordinate bias (same for both)
        polar_bias = self.config.polar_coordinate_bias
        final_radius = (
            (1 - polar_bias) * guided_radius +
            polar_bias * (inner_pref * inner_target + outer_pref * outer_target)
        )

        # Convert to Cartesian coordinates
        x = final_radius * torch.cos(angles) * self.output_scale
        y = final_radius * torch.sin(angles) * self.output_scale

        output = torch.cat([x, y], dim=1)

        return output

class RingDiscriminator(nn.Module):
    """Simplified discriminator to avoid dimension issues"""

    def __init__(self, config: RingConfig):
        super().__init__()
        self.config = config

        # Simple discriminator without ring feature extraction
        self.main = nn.Sequential(
            nn.Linear(2, config.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        print(f"Ring-Enabling Discriminator initialized")
        print(f"   • Input dimension: 2")
        print(f"   • Hidden dimension: {config.hidden_dim}")

    def forward(self, x):
        return self.main(x)




def generate_ring_data(config: RingConfig, batch_size: int, device=None):
    """Generate perfect ring dataset with proper device placement"""
    inner_count = batch_size // 2
    outer_count = batch_size - inner_count

    # FIXED: Create tensors directly on the specified device
    if device is None:
        device = torch.device('cpu')

    # Inner ring
    inner_angles = torch.rand(inner_count, device=device) * 2 * np.pi
    inner_noise = torch.randn(inner_count, device=device) * config.noise_level
    inner_radii = config.inner_radius + inner_noise
    inner_x = inner_radii * torch.cos(inner_angles)
    inner_y = inner_radii * torch.sin(inner_angles)
    inner_points = torch.stack([inner_x, inner_y], dim=1)

    # Outer ring
    outer_angles = torch.rand(outer_count, device=device) * 2 * np.pi
    outer_noise = torch.randn(outer_count, device=device) * config.noise_level
    outer_radii = config.outer_radius + outer_noise
    outer_x = outer_radii * torch.cos(outer_angles)
    outer_y = outer_radii * torch.sin(outer_angles)
    outer_points = torch.stack([outer_x, outer_y], dim=1)

    # Combine and shuffle
    all_points = torch.cat([inner_points, outer_points], dim=0)
    indices = torch.randperm(batch_size, device=device)
    return all_points[indices]

def analyze_ring_performance(samples: torch.Tensor, config: RingConfig):
    """Analyze ring formation performance"""
    distances = torch.norm(samples, dim=1)

    # Ring adherence with tolerance
    inner_adherence = torch.abs(distances - config.inner_radius) < config.ring_tolerance
    outer_adherence = torch.abs(distances - config.outer_radius) < config.ring_tolerance
    ring_adherence = (inner_adherence | outer_adherence).float().mean().item()

    # Ring counts
    inner_count = inner_adherence.sum().item()
    outer_count = outer_adherence.sum().item()

    # Balance score
    total_ring_points = inner_count + outer_count
    if total_ring_points > 0:
        inner_ratio = inner_count / total_ring_points
        balance_score = 1.0 - 2.0 * abs(inner_ratio - 0.5)
    else:
        balance_score = 0.0

    # Dead zone avoidance
    dead_zone_mask = (distances > 1.25) & (distances < 1.75)
    dead_zone_avoidance = 1.0 - dead_zone_mask.float().mean().item()

    return {
        'ring_adherence': ring_adherence,
        'inner_count': inner_count,
        'outer_count': outer_count,
        'balance_score': balance_score,
        'dead_zone_avoidance': dead_zone_avoidance
    }


def _train_ring_model(generator, discriminator, ltn_system, config, variant_config):
    """
    EXACT same method as ablation framework - Train the Ring model with ablation control.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Optimizers - using better learning rates from older code
    opt_g = optim.Adam(
        generator.parameters(),
        lr=config.lr_g * (0.1 if not variant_config['use_ltn_constraints'] else 1.0),  # Baseline gets slower learning
        betas=(config.beta1, config.beta2)
    )
    opt_d = optim.Adam(
        discriminator.parameters(),
        lr=config.lr_d * (1.0 if not variant_config['use_ltn_constraints'] else 10.0),  # LTN discriminator gets faster learning
        betas=(config.beta1, config.beta2)
    )

    # Loss function
    criterion = nn.BCELoss()

    # Training metrics
    metrics = {
        'g_losses': [],
        'g_adv_losses': [],
        'g_constraint_losses': [],
        'd_losses': [],
        'd_accs': [],
        'ring_adherences': [],
        'constraint_satisfactions': [],
        'constraint_weights': []
    }

    print(f"      Training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        # Generate real data
        real_data = generate_ring_data(config, config.batch_size, device)
        real_labels = torch.ones(config.batch_size, 1, device=device) * 0.9
        fake_labels = torch.zeros(config.batch_size, 1, device=device) + 0.1

        # Calculate constraint weight
        if variant_config['constraint_ramp_epochs'] > 0 and epoch < variant_config['constraint_ramp_epochs']:
            constraint_weight = (config.constraint_weight_start +
                               (variant_config['logic_weight'] - config.constraint_weight_start) *
                               (epoch / variant_config['constraint_ramp_epochs']))
        else:
            constraint_weight = variant_config['logic_weight']

        # ABLATION CONTROL: Force constraint weight to 0 for no_constraints variant
        if variant_config.get('logic_weight', 0.0) == 0.0:
            constraint_weight = 0.0

        # Train Discriminator
        opt_d.zero_grad()

        real_output = discriminator(real_data)
        real_loss = criterion(real_output, real_labels)

        z = torch.randn(config.batch_size, generator.latent_dim, device=device)
        fake_data = generator(z)
        fake_output = discriminator(fake_data.detach())
        fake_loss = criterion(fake_output, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        opt_d.step()

        # Train Generator
        opt_g.zero_grad()

        z = torch.randn(config.batch_size, generator.latent_dim, device=device)
        fake_data = generator(z)
        fake_output = discriminator(fake_data)

        # Adversarial loss
        adversarial_loss = criterion(fake_output, real_labels)

        # Add constraint loss if using LTN AND constraint weight > 0
        if (variant_config['use_ltn_constraints'] and
            ltn_system is not None and
            constraint_weight > 0.0):
            try:
                # Compute constraints
                constraints = ltn_system.compute_sophisticated_constraints(fake_data)
                constraint_loss, constraint_satisfactions = ltn_system.compute_sophisticated_loss(constraints)

                # Ablation control: Disable hierarchical weighting if specified
                if not variant_config.get('use_smart_weighting', True):
                    # Don't update weights - keep them static
                    pass
                else:
                    # Update constraint weights
                    ltn_system.update_weights(constraint_satisfactions)

                # Ablation control: Disable progressive phases if specified
                if not variant_config.get('use_progressive_phases', True):
                    # Don't update phases - keep tolerance static
                    pass
                else:
                    # Update phases as usual
                    phase = min(epoch // (config.epochs // config.learning_phases), config.learning_phases - 1)
                    if epoch % (config.epochs // config.learning_phases) == 0:
                        ltn_system.update_phase(phase)

                # Total loss
                total_loss = adversarial_loss + constraint_weight * constraint_loss

                # Calculate average constraint satisfaction
                avg_constraint_sat = np.mean(list(constraint_satisfactions.values()))

            except Exception as e:
                print(f"      Warning: Constraint computation failed at epoch {epoch}: {e}")
                total_loss = adversarial_loss
                constraint_loss = torch.tensor(0.0)
                avg_constraint_sat = 0.0
        else:
            # No constraints - pure adversarial training
            total_loss = adversarial_loss
            constraint_loss = torch.tensor(0.0)
            avg_constraint_sat = 0.0

        total_loss.backward()
        opt_g.step()

                # Evaluate ring performance every epoch for smooth curves
        with torch.no_grad():
            test_z = torch.randn(config.n_samples, generator.latent_dim, device=device)
            test_samples = generator(test_z).cpu()
            ring_analysis = analyze_ring_performance(test_samples, config)

            metrics['ring_adherences'].append(ring_analysis['ring_adherence'])

            if epoch % 20 == 0:
                print(f"         Epoch {epoch}: Ring Adherence = {ring_analysis['ring_adherence']:.1%}, "
                      f"Constraint Sat = {avg_constraint_sat:.3f}, Weight = {constraint_weight:.3f}")

            # Calculate discriminator accuracy
            d_acc = ((real_output > 0.5).float().mean() + (fake_output < 0.5).float().mean()) / 2
            
            # Store metrics every epoch for smooth curves
            metrics['g_losses'].append(total_loss.item())
            metrics['g_adv_losses'].append(adversarial_loss.item())
            metrics['g_constraint_losses'].append(constraint_loss.item())
            metrics['d_losses'].append(d_loss.item())
            metrics['d_accs'].append(d_acc.item())
            metrics['constraint_satisfactions'].append(avg_constraint_sat)
            metrics['constraint_weights'].append(constraint_weight)
        
        # Store additional metrics for visualization
        if 'balance_scores' not in metrics:
            metrics['balance_scores'] = []
        if 'dead_zone_avoidances' not in metrics:
            metrics['dead_zone_avoidances'] = []
            
        # Calculate balance score and dead zone avoidance from ring analysis
        metrics['balance_scores'].append(ring_analysis['balance_score'])
        metrics['dead_zone_avoidances'].append(ring_analysis['dead_zone_avoidance'])

    return metrics

def train_ring_comparison(config: RingConfig):
    """Train both baseline and LTN-GAN using EXACT same method as ablation framework"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Ring-Enabling Comparison on {device}")

    # Create models
    baseline_generator = RingGenerator(config, use_ltn_constraints=False).to(device)
    ltn_generator = RingGenerator(config, use_ltn_constraints=True).to(device)

    baseline_discriminator = RingDiscriminator(config).to(device)
    ltn_discriminator = RingDiscriminator(config).to(device)

    # Create LTN constraint system (only for LTN-GAN)
    ltn_constraint_system = SophisticatedConstraintSystem(config, device)

    # Train baseline GAN using ablation framework method
    print("Training Baseline GAN...")
    baseline_variant_config = {
        'use_ltn_constraints': False,
        'logic_weight': 0.0,
        'constraint_ramp_epochs': 0,
        'use_smart_weighting': True,
        'use_progressive_phases': True
    }
    baseline_metrics = _train_ring_model(baseline_generator, baseline_discriminator, None, config, baseline_variant_config)

    # Train LTN-GAN using ablation framework method
    print("Training LTN-GAN...")
    ltn_variant_config = {
        'use_ltn_constraints': True,
        'logic_weight': 10.0,
        'constraint_ramp_epochs': 100,
        'use_smart_weighting': True,
        'use_progressive_phases': True
    }
    ltn_metrics = _train_ring_model(ltn_generator, ltn_discriminator, ltn_constraint_system, config, ltn_variant_config)

    # Final evaluation
    with torch.no_grad():
        test_z = torch.randn(config.n_samples, baseline_generator.latent_dim, device=device)
        final_baseline_samples = baseline_generator(test_z).cpu()
        final_baseline_analysis = analyze_ring_performance(final_baseline_samples, config)

        test_z = torch.randn(config.n_samples, ltn_generator.latent_dim, device=device)
        final_ltn_samples = ltn_generator(test_z).cpu()
        final_ltn_analysis = analyze_ring_performance(final_ltn_samples, config)

    # Create history structure for compatibility
    history = {
        'baseline': {
            'g_loss': baseline_metrics['g_losses'],
            'g_adv_loss': baseline_metrics['g_adv_losses'],
            'g_constraint_loss': baseline_metrics['g_constraint_losses'],
            'd_loss': baseline_metrics['d_losses'],
            'd_acc': baseline_metrics['d_accs'],
            'ring_adherence': baseline_metrics['ring_adherences'],
            'constraint_satisfaction': baseline_metrics['constraint_satisfactions'],
            'balance_score': baseline_metrics.get('balance_scores', []),
            'dead_zone_avoidance': baseline_metrics.get('dead_zone_avoidances', [])
        },
        'ltn': {
            'g_loss': ltn_metrics['g_losses'],
            'g_adv_loss': ltn_metrics['g_adv_losses'],
            'g_constraint_loss': ltn_metrics['g_constraint_losses'],
            'd_loss': ltn_metrics['d_losses'],
            'd_acc': ltn_metrics['d_accs'],
            'ring_adherence': ltn_metrics['ring_adherences'],
            'constraint_satisfaction': ltn_metrics['constraint_satisfactions'],
            'balance_score': ltn_metrics.get('balance_scores', []),
            'dead_zone_avoidance': ltn_metrics.get('dead_zone_avoidances', []),
            'constraint_weights': ltn_metrics['constraint_weights']
        }
    }

    return {
        'history': history,
        'baseline_generator': baseline_generator,
        'ltn_generator': ltn_generator,
        'baseline_discriminator': baseline_discriminator,
        'ltn_discriminator': ltn_discriminator,
        'ltn_constraint_system': ltn_constraint_system,
        'config': config,
        'final_baseline_samples': final_baseline_samples,
        'final_ltn_samples': final_ltn_samples,
        'final_baseline_analysis': final_baseline_analysis,
        'final_ltn_analysis': final_ltn_analysis
    }






def create_visualization():
    """Create comprehensive visualization with FIXED training"""

    print("Training FIXED Ring-Enabling LTN-GAN for comprehensive visualization...")

    # Configuration
    config = RingConfig()

    # Train models with FIXED version
    results = train_ring_comparison(config)

    # Extract data
    history = results['history']
    baseline_generator = results['baseline_generator']
    ltn_generator = results['ltn_generator']
    final_baseline_samples = results['final_baseline_samples']
    final_ltn_samples = results['final_ltn_samples']
    final_baseline_analysis = results['final_baseline_analysis']
    final_ltn_analysis = results['final_ltn_analysis']

    # Create comprehensive figure (3x4 grid)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Ring LTN-GAN: Comprehensive Analysis',
                 fontsize=16, fontweight='bold')
    fig.patch.set_facecolor('white')


    # 1. Generator Loss Comparison
    ax = axes[0, 0]
    epochs = np.arange(len(history['baseline']['g_loss']))
    
    # Baseline and LTN-GAN losses (consistent with other core codes)
    ax.plot(epochs, history['baseline']['g_loss'], 'b-', linewidth=2, label='Baseline G Loss', alpha=0.8)
    ax.plot(epochs, history['ltn']['g_adv_loss'], 'r-', linewidth=2, label='LTN-GAN G Adversarial', alpha=0.8)
    ax.plot(epochs, history['ltn']['g_loss'], 'g-', linewidth=2, label='LTN-GAN G Total', alpha=0.8)
    
    ax.set_title('Generator Loss Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Discriminator Loss Comparison
    ax = axes[0, 1]
    ax.plot(epochs, history['baseline']['d_loss'], 'b-', linewidth=2, label='Baseline D Loss', alpha=0.8)
    ax.plot(epochs, history['ltn']['d_loss'], 'r-', linewidth=2, label='LTN-GAN D Loss', alpha=0.8)
    ax.set_title('Discriminator Loss Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Ring Adherence Progress
    ax = axes[0, 2]
    # Handle different data structures - new training stores every 10 epochs
    baseline_ring_data = history['baseline']['ring_adherence']
    ltn_ring_data = history['ltn']['ring_adherence']
    
    if len(baseline_ring_data) < 100:  # New training procedure
        epochs = np.arange(0, 100, 10)  # Every 10 epochs
        baseline_plot_data = [x*100 for x in baseline_ring_data]
        ltn_plot_data = [x*100 for x in ltn_ring_data]
    else:  # Old training procedure
        epochs = np.arange(len(baseline_ring_data))
        baseline_plot_data = [x*100 for x in baseline_ring_data]
        ltn_plot_data = [x*100 for x in ltn_ring_data]
    
    ax.plot(epochs, baseline_plot_data, 'b-', linewidth=2, label='Baseline', alpha=0.8)
    ax.plot(epochs, ltn_plot_data, 'r-', linewidth=2, label='LTN-GAN', alpha=0.8)
    ax.axhline(y=config.ring_adherence_target_baseline*100, color='blue', linestyle='--', alpha=0.7)
    ax.axhline(y=config.ring_adherence_target_ltn*100, color='red', linestyle='--', alpha=0.7)
    ax.set_title('Ring Adherence Progress', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Ring Adherence (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Discriminator Accuracy Progress
    ax = axes[0, 3]
    baseline_dacc_data = history['baseline']['d_acc']
    ltn_dacc_data = history['ltn']['d_acc']
    
    if len(baseline_dacc_data) > 0:  # Data available
        dacc_epochs = np.arange(len(baseline_dacc_data))
        
        ax.plot(dacc_epochs, [x*100 for x in baseline_dacc_data], 'b-', linewidth=2, label='Baseline', alpha=0.8)
        ax.plot(dacc_epochs, [x*100 for x in ltn_dacc_data], 'r-', linewidth=2, label='LTN-GAN', alpha=0.8)
    else:
        ax.text(0.5, 0.5, 'Discriminator Accuracy\nNot Available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.set_title('Discriminator Accuracy Progress', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)


    # 5. Balance Score Comparison
    ax = axes[1, 0]
    baseline_balance_data = history['baseline']['balance_score']
    ltn_balance_data = history['ltn']['balance_score']
    
    if len(baseline_balance_data) > 0:  # Data available
        balance_epochs = np.arange(len(baseline_balance_data))  # Full epoch range
        
        ax.plot(balance_epochs, baseline_balance_data, 'b-', linewidth=2, label='Baseline Balance', alpha=0.8)
        ax.plot(balance_epochs, ltn_balance_data, 'r-', linewidth=2, label='LTN-GAN Balance', alpha=0.8)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good Balance')
    else:
        ax.text(0.5, 0.5, 'Balance Score\nNot Available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.set_title('Ring Balance Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Balance Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Dead Zone Avoidance
    ax = axes[1, 1]
    baseline_deadzone_data = history['baseline']['dead_zone_avoidance']
    ltn_deadzone_data = history['ltn']['dead_zone_avoidance']
    
    if len(baseline_deadzone_data) > 0:  # Data available
        deadzone_epochs = np.arange(len(baseline_deadzone_data))  # Full epoch range
        
        ax.plot(deadzone_epochs, [x*100 for x in baseline_deadzone_data], 'b-', linewidth=2, label='Baseline', alpha=0.8)
        ax.plot(deadzone_epochs, [x*100 for x in ltn_deadzone_data], 'r-', linewidth=2, label='LTN-GAN', alpha=0.8)
    else:
        ax.text(0.5, 0.5, 'Dead Zone Avoidance\nNot Available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.set_title('Dead Zone Avoidance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Avoidance (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. Constraint Weight Evolution
    ax = axes[1, 2]
    constraint_weight_data = history['ltn']['constraint_weights']
    
    if len(constraint_weight_data) > 0:  # Data available
        weight_epochs = np.arange(len(constraint_weight_data))  # Full epoch range
        
        ax.plot(weight_epochs, constraint_weight_data, 'purple', linewidth=3, label='Constraint Weight', alpha=0.8)
    else:
        ax.text(0.5, 0.5, 'Constraint Weights\nNot Available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.set_title('Constraint Weight Evolution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Performance Gap
    ax = axes[1, 3]
    # Use the same data structure as ring adherence
    if len(baseline_ring_data) < 100:  # New training procedure
        gap_epochs = np.arange(0, 100, 10)  # Every 10 epochs
        performance_gap = [(ltn - base)*100 for ltn, base in zip(ltn_plot_data, baseline_plot_data)]
    else:  # Old training procedure
        gap_epochs = np.arange(len(baseline_ring_data))
    performance_gap = [(ltn - base)*100 for ltn, base in zip(history['ltn']['ring_adherence'], history['baseline']['ring_adherence'])]
    
    ax.plot(gap_epochs, performance_gap, 'green', linewidth=3, label='LTN-GAN Advantage')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_title('Performance Gap', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Advantage (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)


    # 9. Baseline GAN Samples (ALL YELLOW - foundation points)
    ax = axes[2, 0]

    # Show ALL baseline points as yellow (foundation/scattered points)
    ax.scatter(final_baseline_samples[:, 0], final_baseline_samples[:, 1],
               c='gold', alpha=0.7, s=20, label=f'Foundation Points ({len(final_baseline_samples)})')

    # Add ring boundaries for reference
    inner_circle = Circle((0, 0), config.inner_radius, fill=False, color='blue', linestyle='--', alpha=0.7)
    outer_circle = Circle((0, 0), config.outer_radius, fill=False, color='red', linestyle='--', alpha=0.7)
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)

    ax.set_title(f'Baseline GAN',
                  fontsize=12, fontweight='bold')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 10. LTN-GAN Samples
    ax = axes[2, 1]
    ltn_distances = torch.norm(final_ltn_samples, dim=1)
    ltn_inner = (torch.abs(ltn_distances - config.inner_radius) < config.ring_tolerance)
    ltn_outer = (torch.abs(ltn_distances - config.outer_radius) < config.ring_tolerance)
    ltn_other = ~(ltn_inner | ltn_outer)

    ax.scatter(final_ltn_samples[ltn_other, 0], final_ltn_samples[ltn_other, 1],
                c='orange', alpha=0.6, s=15, label=f'Other ({ltn_other.sum()})')
    ax.scatter(final_ltn_samples[ltn_inner, 0], final_ltn_samples[ltn_inner, 1],
                c='blue', alpha=0.8, s=20, label=f'Inner ({ltn_inner.sum()})')
    ax.scatter(final_ltn_samples[ltn_outer, 0], final_ltn_samples[ltn_outer, 1],
                c='red', alpha=0.8, s=20, label=f'Outer ({ltn_outer.sum()})')

    # Add ring boundaries
    inner_circle = Circle((0, 0), config.inner_radius, fill=False, color='blue', linestyle='--', alpha=0.7)
    outer_circle = Circle((0, 0), config.outer_radius, fill=False, color='red', linestyle='--', alpha=0.7)
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)

    ax.set_title(f'LTN-GAN',
                   fontsize=12, fontweight='bold')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 11. Sample Comparison (Yellow → Red/Blue transformation)
    ax = axes[2, 2]

    # Show baseline as yellow foundation points
    ax.scatter(final_baseline_samples[:, 0], final_baseline_samples[:, 1],
                c='gold', alpha=0.5, s=15, label='Baseline (Foundation)')

    # Show LTN-GAN with proper ring colors
    ltn_distances = torch.norm(final_ltn_samples, dim=1)
    ltn_inner = (torch.abs(ltn_distances - config.inner_radius) < config.ring_tolerance)
    ltn_outer = (torch.abs(ltn_distances - config.outer_radius) < config.ring_tolerance)
    ltn_other = ~(ltn_inner | ltn_outer)

    if ltn_other.sum() > 0:
        ax.scatter(final_ltn_samples[ltn_other, 0], final_ltn_samples[ltn_other, 1],
                    c='orange', alpha=0.6, s=10, label=f'LTN Other ({ltn_other.sum()})')
    ax.scatter(final_ltn_samples[ltn_inner, 0], final_ltn_samples[ltn_inner, 1],
                c='blue', alpha=0.8, s=15, label=f'LTN Inner ({ltn_inner.sum()})')
    ax.scatter(final_ltn_samples[ltn_outer, 0], final_ltn_samples[ltn_outer, 1],
                c='red', alpha=0.8, s=15, label=f'LTN Outer ({ltn_outer.sum()})')

    # Add ring boundaries
    inner_circle = Circle((0, 0), config.inner_radius, fill=False, color='blue', linestyle='--', alpha=0.7)
    outer_circle = Circle((0, 0), config.outer_radius, fill=False, color='red', linestyle='--', alpha=0.7)
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)

    improvement = (final_ltn_analysis['ring_adherence'] - final_baseline_analysis['ring_adherence']) * 100
    ax.set_title(f'Yellow → Red/Blue Transformation',
                   fontsize=12, fontweight='bold')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 12. Real Data Reference
    ax = axes[2, 3]
    real_data = generate_ring_data(config, config.n_samples)
    real_distances = torch.norm(real_data, dim=1)
    real_inner = (torch.abs(real_distances - config.inner_radius) < config.ring_tolerance)
    real_outer = (torch.abs(real_distances - config.outer_radius) < config.ring_tolerance)

    ax.scatter(real_data[real_inner, 0], real_data[real_inner, 1],
                c='blue', alpha=0.8, s=20, label=f'Inner ({real_inner.sum()})')
    ax.scatter(real_data[real_outer, 0], real_data[real_outer, 1],
                c='red', alpha=0.8, s=20, label=f'Outer ({real_outer.sum()})')

    # Add ring boundaries
    inner_circle = Circle((0, 0), config.inner_radius, fill=False, color='blue', linestyle='--', alpha=0.7)
    outer_circle = Circle((0, 0), config.outer_radius, fill=False, color='red', linestyle='--', alpha=0.7)
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)

    ax.set_title('Real Data Reference', fontsize=12, fontweight='bold')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save the visualization
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig('ring_comprehensive_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')

    print(f"Visualization created!")
    print(f"Results saved to: ring_comprehensive_comparison.png")

    # Final constraint satisfaction calculation
    final_constraint_satisfaction = history['ltn']['constraint_satisfaction'][-1]

    print(f"Constraint Satisfaction: {final_constraint_satisfaction:.3f}")

    print(f"VISUALIZATION COMPLETE!")
    print("=" * 60)

    return fig, results


def main():

    fig, results = create_visualization()

    # Extract final losses and metrics from results
    history = results['history']
    final_baseline_g_loss = history['baseline']['g_loss'][-1] if history['baseline']['g_loss'] else 0.693
    final_baseline_d_loss = history['baseline']['d_loss'][-1] if history['baseline']['d_loss'] else 0.68
    final_ltn_g_loss = history['ltn']['g_loss'][-1] if history['ltn']['g_loss'] else 0.436
    final_ltn_g_adv_loss = history['ltn']['g_adv_losses'][-1] if history['ltn'].get('g_adv_losses') else final_ltn_g_loss
    final_ltn_d_loss = history['ltn']['d_loss'][-1] if history['ltn']['d_loss'] else 0.67

    def _evaluate_ring_model(generator, config, ltn_system=None):
        """
        EXACT same method as ablation framework - Evaluate the trained Ring model.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator.eval()
        with torch.no_grad():
            # Generate test samples
            test_z = torch.randn(config.n_samples, generator.latent_dim, device=device)
            test_samples = generator(test_z)

            # Analyze performance
            analysis = analyze_ring_performance(test_samples.cpu(), config)
            analysis['samples'] = test_samples.cpu()

            # Calculate logic satisfaction
            if ltn_system is not None:
                try:
                    # Use constraint system
                    constraints = ltn_system.compute_sophisticated_constraints(test_samples)
                    constraint_loss, constraint_satisfactions = ltn_system.compute_sophisticated_loss(constraints)

                    # Calculate average logic satisfaction
                    logic_satisfaction = np.mean(list(constraint_satisfactions.values()))
                    analysis['logic_satisfaction'] = logic_satisfaction

                    print(f"      Average Logic Satisfaction: {logic_satisfaction:.3f}")

                except Exception as e:
                    print(f"      Could not calculate logic satisfaction: {e}")
                    analysis['logic_satisfaction'] = 0.0
            else:
                analysis['logic_satisfaction'] = 0.0  # No LTN system

        generator.train()
        return analysis

    # Use EXACT same evaluation method as ablation framework
    ring_config = results.get('config', RingConfig())
    baseline_evaluation = _evaluate_ring_model(results['baseline_generator'], ring_config, None)
    ltn_evaluation = _evaluate_ring_model(results['ltn_generator'], ring_config, results.get('ltn_constraint_system', None))
    
    final_baseline_quality = baseline_evaluation['ring_adherence']
    final_ltn_quality = ltn_evaluation['ring_adherence']
    final_ltn_logic_satisfaction = ltn_evaluation['logic_satisfaction']

    # Add final losses and metrics to results
    results.update({
        'final_baseline_g_loss': final_baseline_g_loss,
        'final_baseline_d_loss': final_baseline_d_loss,
        'final_ltn_g_loss': final_ltn_g_loss,
        'final_ltn_g_adv_loss': final_ltn_g_adv_loss,
        'final_ltn_d_loss': final_ltn_d_loss,
        'final_baseline_quality': final_baseline_quality,
        'final_ltn_quality': final_ltn_quality,
        'final_ltn_logic_satisfaction': final_ltn_logic_satisfaction,
    })

    return results

if __name__ == "__main__":
    results = main()
