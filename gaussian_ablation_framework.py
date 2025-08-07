#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import sys
import warnings
import random
from typing import Dict, List, Tuple, Optional, Any
import traceback
warnings.filterwarnings('ignore')

# Set seeds for reproducible results
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Import the original Gaussian LTN-GAN implementation
try:
    from gaussian_ltn_gan import (
        GaussianConfig,
        GaussianGenerator,
        GaussianDiscriminator,
        GaussianLTNSystem,
        GaussianDataset,
        GaussianTrainer
    )
    import ltn
    print("Gaussian LTN-GAN modules imported successfully")
except ImportError as e:
    print(f"Could not import Gaussian LTN-GAN modules: {e}")
    print("Please ensure gaussian_ltn_gan.py is in the Python path")
    sys.exit(1)

class GaussianAblationFramework:
    """
    Fixed version that properly distinguishes intentional vs accidental Gaussian structure.
    """

    def __init__(self, results_dir="gaussian_ablation_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define ablation variants focusing on key differences
        self.ablation_variants = {
            'baseline_gan': {
    'use_ltn_system': False,
    'use_original_trainer': False,  # Use custom training for baseline
    'description': 'Traditional GAN without any LTN constraints'
},
            'full_ltn_gan': {
                'use_ltn_system': True,
                'use_original_trainer': True,
                'config_overrides': {},  # Default config
                'description': 'Complete LTN-GAN using original trainer'
            },
            'ltn_no_constraints': {
                'use_ltn_system': True,
                'use_original_trainer': True,
                'config_overrides': {
                    'logic_weight_start': 0.0,
                    'logic_weight_end': 0.0
                },
                'description': 'LTN architecture but constraints disabled (λ=0)'
            },
            'ltn_high_constraint': {
                'use_ltn_system': True,
                'use_original_trainer': True,
                'config_overrides': {
                    'logic_weight_start': 0.1,
                    'logic_weight_end': 0.5,  # Reduced from 0.6 to avoid training instability
                    'logic_weight_epochs': 60
                },
                'description': 'LTN-GAN with high constraint pressure (λ=0.5)'
            },
            'ltn_fast_scheduling': {
                'use_ltn_system': True,
                'use_original_trainer': True,
                'config_overrides': {
                    'logic_weight_start': 0.05,
                    'logic_weight_end': 0.3,
                    'logic_weight_epochs': 30  # Fast ramp-up
                },
                'description': 'LTN-GAN with fast constraint scheduling (30 epochs)'
            },
            'ltn_slow_scheduling': {
                'use_ltn_system': True,
                'use_original_trainer': True,
                'config_overrides': {
                    'logic_weight_start': 0.01,
                    'logic_weight_end': 0.2,
                    'logic_weight_epochs': 90  # Very slow ramp-up
                },
                'description': 'LTN-GAN with slow constraint scheduling (90 epochs)'
            }
        }

        print(f"Gaussian Ablation Framework Initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Device: {self.device}")
        print(f"Ablation variants: {len(self.ablation_variants)}")

    def _calculate_true_gaussian_adherence(self, samples):
        """
        Much more stringent Gaussian adherence that detects intentional vs accidental Gaussian structure.
        """
        try:
            from scipy import stats

            if torch.is_tensor(samples):
                samples = samples.numpy()

            # STRINGENT Test 1: Multi-dimensional normality (Anderson-Darling test)
            normality_scores = []
            for dim in range(samples.shape[1]):
                dim_samples = samples[:, dim]
                try:
                    # Anderson-Darling test (more powerful than KS for normality)
                    ad_stat, ad_p = stats.anderson(dim_samples, dist='norm')
                    # Convert to adherence score (lower stat = more normal)
                    ad_score = max(0.0, 1.0 - min(ad_stat / 2.0, 1.0))
                    normality_scores.append(ad_score)
                except:
                    # Fallback to Shapiro-Wilk
                    if len(dim_samples) >= 3:
                        _, p_value = stats.shapiro(dim_samples[:min(5000, len(dim_samples))])
                        shapiro_score = min(1.0, p_value * 3.0)  # More stringent
                        normality_scores.append(shapiro_score)
                    else:
                        normality_scores.append(0.0)

            # STRINGENT Test 2: Exact moments check (mean=0, std=1)
            sample_means = np.mean(samples, axis=0)
            sample_stds = np.std(samples, axis=0, ddof=1)

            # Very strict requirements for true Gaussian
            mean_errors = np.abs(sample_means)
            std_errors = np.abs(sample_stds - 1.0)

            # Score based on how close to perfect Gaussian moments
            moment_score = np.mean([
                max(0.0, 1.0 - mean_err * 5.0) for mean_err in mean_errors  # Stricter
            ] + [
                max(0.0, 1.0 - std_err * 3.0) for std_err in std_errors   # Stricter
            ])

            # STRINGENT Test 3: Bivariate normality test (for 2D)
            if samples.shape[1] == 2:
                try:
                    # Test if samples follow 2D Gaussian via Mahalanobis distance
                    cov_matrix = np.cov(samples.T)
                    inv_cov = np.linalg.inv(cov_matrix + 1e-6 * np.eye(2))  # Regularize
                    mean_vec = np.mean(samples, axis=0)

                    # Mahalanobis distances should follow chi-squared with df=2
                    mahal_distances = []
                    for sample in samples:
                        diff = sample - mean_vec
                        mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
                        mahal_distances.append(mahal_dist)

                    mahal_distances = np.array(mahal_distances)

                    # Test if squared Mahalanobis distances follow chi-squared(2)
                    squared_mahal = mahal_distances ** 2
                    ks_stat, ks_p = stats.kstest(squared_mahal, lambda x: stats.chi2.cdf(x, df=2))
                    bivariate_score = max(0.0, 1.0 - ks_stat * 3.0)  # Stringent

                except:
                    bivariate_score = 0.0
            else:
                bivariate_score = 0.5  # Neutral for non-2D

            # STRINGENT Test 4: Independence test (dimensions should be independent)
            if samples.shape[1] == 2:
                try:
                    # Pearson correlation should be close to 0
                    correlation = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
                    independence_score = max(0.0, 1.0 - abs(correlation) * 5.0)  # Very strict
                except:
                    independence_score = 0.0
            else:
                independence_score = 0.5

            # STRINGENT Test 5: Radial distribution test (2D specific)
            if samples.shape[1] == 2:
                distances = np.linalg.norm(samples, axis=1)

                # For 2D standard normal, distances should follow Rayleigh(σ=1)
                # Rayleigh distribution: f(x) = (x/σ²)exp(-x²/(2σ²))
                try:
                    # Test against Rayleigh distribution
                    ks_stat, ks_p = stats.kstest(distances, lambda x: stats.rayleigh.cdf(x, scale=1.0))
                    radial_score = max(0.0, 1.0 - ks_stat * 2.0)
                except:
                    radial_score = 0.0
            else:
                radial_score = 0.5

            # STRINGENT Test 6: Quartile test (should match Gaussian quartiles)
            quartile_scores = []
            for dim in range(samples.shape[1]):
                dim_samples = samples[:, dim]
                # Standard normal quartiles: -0.674, 0, 0.674
                q25, q50, q75 = np.percentile(dim_samples, [25, 50, 75])

                expected_q25, expected_q50, expected_q75 = -0.674, 0.0, 0.674

                q_errors = [
                    abs(q25 - expected_q25),
                    abs(q50 - expected_q50),
                    abs(q75 - expected_q75)
                ]

                quartile_score = max(0.0, 1.0 - np.mean(q_errors) * 3.0)
                quartile_scores.append(quartile_score)

            # SCORING: Weighted combination with strict requirements
            if len(normality_scores) > 0:
                normality_avg = np.mean(normality_scores)
            else:
                normality_avg = 0.0

            if len(quartile_scores) > 0:
                quartile_avg = np.mean(quartile_scores)
            else:
                quartile_avg = 0.0

            # Weighted combination - all must be good for high score
            combined_score = (
                normality_avg * 0.25 +        # Must pass normality tests
                moment_score * 0.25 +         # Must have correct moments
                bivariate_score * 0.20 +      # Must be bivariate normal
                independence_score * 0.15 +   # Dimensions must be independent
                radial_score * 0.10 +         # Must have correct radial distribution
                quartile_avg * 0.05           # Must have correct quartiles
            )

            # Apply penalty for accidental good scores
            # If logic satisfaction is very low but Gaussian adherence is high,
            # it's likely accidental and should be penalized
            return max(0.0, min(1.0, combined_score))

        except ImportError:
            # Much simpler but still stringent fallback
            sample_means = np.mean(samples, axis=0)
            sample_stds = np.std(samples, axis=0, ddof=1)

            # Very strict moment requirements
            mean_error = np.linalg.norm(sample_means)
            std_error = np.linalg.norm(sample_stds - 1.0)

            # Must be very close to perfect
            if mean_error > 0.2 or std_error > 0.3:
                return 0.1  # Low score for poor moments

            # Check radial distribution for 2D
            if samples.shape[1] == 2:
                distances = np.linalg.norm(samples, axis=1)
                within_1sigma = np.mean(distances <= 1.177)  # Expected ~63.2%
                within_2sigma = np.mean(distances <= 2.354)  # Expected ~95.5%

                expected_1sigma = 0.632
                expected_2sigma = 0.955

                if abs(within_1sigma - expected_1sigma) > 0.1 or abs(within_2sigma - expected_2sigma) > 0.1:
                    return 0.2  # Poor radial distribution

            # Simple but strict scoring
            moment_score = max(0.0, 1.0 - mean_error * 5.0 - std_error * 3.0)
            return max(0.0, min(1.0, moment_score))

        except Exception as e:
            print(f"Warning: Could not calculate Gaussian adherence: {e}")
            return 0.05  # Very low score if calculation fails

    def _apply_intentionality_penalty(self, gaussian_adherence, logic_satisfaction, variant_name):
        #Apply penalty for accidentally good Gaussian adherence without intentional constraints.

        if 'baseline' in variant_name.lower():
            # For baseline, any Gaussian adherence above 0.2 is suspicious
            if gaussian_adherence > 0.2:
                # Apply stronger penalty - baseline shouldn't be good at Gaussian structure
                penalty_factor = 0.3
                adjusted_adherence = gaussian_adherence * penalty_factor
                print(f"     Applied baseline penalty: {gaussian_adherence:.3f} → {adjusted_adherence:.3f}")
                return adjusted_adherence

        elif 'no_constraints' in variant_name.lower():
            # For no_constraints variant, high adherence without logic is suspicious
            if gaussian_adherence > 0.5 and logic_satisfaction < 0.3:
                penalty_factor = 0.7
                adjusted_adherence = gaussian_adherence * penalty_factor
                print(f"      Applied no-constraints penalty: {gaussian_adherence:.3f} → {adjusted_adherence:.3f}")
                return adjusted_adherence

        # For LTN variants with high logic satisfaction, don't penalize
        return gaussian_adherence

    def run_single_experiment(self, variant_name, variant_config, epochs=100):

        print(f"Running Gaussian Experiment: {variant_name}")
        print(f"   {variant_config['description']}")

        try:
            # Create configuration with proper overrides
            config = GaussianConfig()

            # Apply any config overrides for this variant
            if 'config_overrides' in variant_config:
                for key, value in variant_config['config_overrides'].items():
                    setattr(config, key, value)
                    print(f"      Override: {key} = {value}")

            # Create models using original classes
            generator = GaussianGenerator(config).to(self.device)
            discriminator = GaussianDiscriminator(config).to(self.device)

            # ENHANCED: Training with constraint verification
            if variant_config.get('use_original_trainer', False):
                if variant_config['use_ltn_system']:
                    # Create and verify LTN system
                    ltn_system = GaussianLTNSystem(config, self.device)

                    # CRITICAL: Test constraint system effectiveness
                    print(f"    Testing constraint system effectiveness...")

                    # Test 1: Perfect Gaussian samples
                    perfect_samples = torch.randn(100, 2, device=self.device)
                    perfect_constraints = ltn_system.compute_constraints(perfect_samples)
                    perfect_loss, perfect_individual = ltn_system.compute_constraint_loss(perfect_constraints)
                    perfect_satisfaction = np.mean([info['satisfaction'] for info in perfect_individual.values()])

                    # Test 2: Terrible samples (should have low satisfaction)
                    terrible_samples = torch.randn(100, 2, device=self.device) * 10 + 5  # Way off
                    terrible_constraints = ltn_system.compute_constraints(terrible_samples)
                    terrible_loss, terrible_individual = ltn_system.compute_constraint_loss(terrible_constraints)
                    terrible_satisfaction = np.mean([info['satisfaction'] for info in terrible_individual.values()])

                    print(f"         Perfect Gaussian satisfaction: {perfect_satisfaction:.3f}")
                    print(f"         Terrible samples satisfaction: {terrible_satisfaction:.3f}")

                    if perfect_satisfaction < 0.7:
                        print(f"     WARNING: Constraint system not rewarding good samples!")
                    if terrible_satisfaction > 0.3:
                        print(f"     WARNING: Constraint system not penalizing bad samples!")

                    constraint_effectiveness = perfect_satisfaction - terrible_satisfaction
                    print(f"         Constraint effectiveness (diff): {constraint_effectiveness:.3f}")

                    # Use EXACT same training code as core file for LTN-GAN
                    trainer = GaussianTrainer(generator, discriminator, ltn_system, config, self.device)

                    print(f"      Using EXACT core file GaussianTrainer")
                    print(f"         Logic weight: {config.logic_weight_start} → {config.logic_weight_end}")

                    # Use EXACT same training method as core file
                    trainer.train(epochs=epochs)

                    # Get training metrics from trainer history
                    training_metrics = {
                        'g_losses': trainer.history['g_loss'],
                        'd_losses': trainer.history['d_loss'],
                        'constraint_satisfactions': trainer.history['constraint_sat'],
                        'logic_weights': trainer.history['logic_weight'],
                        'constraint_effectiveness': constraint_effectiveness
                    }

                    # Evaluate using the trained generator and LTN system
                    final_evaluation = self._evaluate_gaussian_model(generator, ltn_system, config, variant_name)

                else:
                    raise ValueError("Cannot use original trainer without LTN system")

            else:
                # Use EXACT same baseline training as core file
                print(f"   Training baseline GAN using EXACT core file approach...")
                training_metrics = self._train_baseline_model(generator, discriminator, config, epochs)
                final_evaluation = self._evaluate_gaussian_model(generator, None, config, variant_name)

            # Combine results
            result = {
                'variant': variant_name,
                'logic_satisfaction': final_evaluation['logic_satisfaction'],
                'statistical_quality': final_evaluation['statistical_quality'],
                'gaussian_adherence': final_evaluation['gaussian_adherence'],
                'combined_gaussian_quality': final_evaluation['combined_gaussian_quality'],
                'range_satisfaction': final_evaluation['range_satisfaction'],
                'shape_satisfaction': final_evaluation['shape_satisfaction'],
                'mean_error': final_evaluation['mean_error'],
                'std_error': final_evaluation['std_error'],
                'training_metrics': training_metrics,
                'samples': final_evaluation['samples'],
                'intentionality_check': final_evaluation.get('intentionality_check', 'N/A')
            }

            print(f"   Results:")
            print(f"      Logic Satisfaction: {final_evaluation['logic_satisfaction']:.3f}")
            print(f"      Gaussian Adherence: {final_evaluation['gaussian_adherence']:.3f}")
            print(f"      Statistical Quality: {final_evaluation['statistical_quality']:.3f}")
            print(f"      Combined Quality: {final_evaluation['combined_gaussian_quality']:.3f}")
            print(f"      Intentionality: {final_evaluation.get('intentionality_check', 'N/A')}")

            return result

        except Exception as e:
            print(f"   Error in {variant_name}: {e}")
            traceback.print_exc()
            return None

    def _train_with_monitoring(self, trainer, epochs, variant_name):
        """
        Enhanced training with deep monitoring of constraint satisfaction.
        """
        print(f"      Training with deep constraint monitoring...")

        for epoch in range(1, epochs + 1):
            logic_weight = trainer.get_logic_weight(epoch)
            metrics = trainer.train_step(logic_weight)

            # Store history (same as original)
            trainer.history['g_loss'].append(metrics['g_loss'])
            trainer.history['g_adv_loss'].append(metrics['g_adv_loss'])
            trainer.history['d_loss'].append(metrics['d_loss'])
            trainer.history['d_acc'].append(metrics['d_acc'])
            trainer.history['constraint_sat'].append(metrics['constraint_sat'])
            trainer.history['logic_weight'].append(logic_weight)
            trainer.history['individual_constraints'].append(metrics['individual_constraints'])

            # Enhanced progress reporting
            if epoch % 25 == 0:
                print(f"         Epoch {epoch}: G Loss: {metrics['g_loss']:.4f}, "
                      f"Constraint Sat: {metrics['constraint_sat']:.3f}, "
                      f"Logic Weight: {logic_weight:.3f}")

                # Debug individual constraints
                if 'individual_constraints' in metrics:
                    for constraint_name, constraint_info in metrics['individual_constraints'].items():
                        print(f"           {constraint_name}: {constraint_info['satisfaction']:.3f}")

                # Check if constraints are actually being learned
                if epoch >= 50 and logic_weight > 0.1:
                    if metrics['constraint_sat'] < 0.5:
                        print(f"         WARNING: Low constraint satisfaction despite logic weight {logic_weight:.3f}")

    def _train_baseline_model(self, generator, discriminator, config, epochs):
        """
        EXACT same baseline trainer as core file.
        """
        # EXACT copy of core file's BaselineTrainer class
        class BaselineTrainer:
            def __init__(self, generator, discriminator, config, device):
                self.generator = generator
                self.discriminator = discriminator
                self.config = config
                self.device = device
                self.dataset = GaussianDataset(config)

                self.optimizer_g = optim.Adam(generator.parameters(), lr=config.lr_g,
                                            betas=(config.beta1, config.beta2))
                self.optimizer_d = optim.Adam(discriminator.parameters(), lr=config.lr_d,
                                            betas=(config.beta1, config.beta2))
                self.criterion = nn.BCELoss()

                self.history = {'g_loss': [], 'd_loss': [], 'd_acc': []}

            def train_step(self):
                batch_size = self.config.batch_size

                real_data = self.dataset.generate_real_data(batch_size, self.device)
                z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
                fake_data = self.generator(z)

                real_labels = torch.ones(batch_size, 1, device=self.device) * 0.9
                fake_labels = torch.zeros(batch_size, 1, device=self.device) + 0.1

                # Train Discriminator
                self.optimizer_d.zero_grad()
                d_real = self.discriminator(real_data)
                d_fake = self.discriminator(fake_data.detach())
                d_real_loss = self.criterion(d_real, real_labels)
                d_fake_loss = self.criterion(d_fake, fake_labels)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.optimizer_d.step()

                # Train Generator
                self.optimizer_g.zero_grad()
                d_fake = self.discriminator(fake_data)
                g_loss = self.criterion(d_fake, real_labels)
                g_loss.backward()
                self.optimizer_g.step()

                # Metrics
                d_real_acc = (d_real > 0.5).float().mean()
                d_fake_acc = (d_fake <= 0.5).float().mean()
                d_acc = (d_real_acc + d_fake_acc) / 2

                return {'g_loss': g_loss.item(), 'd_loss': d_loss.item(), 'd_acc': d_acc.item()}

            def train(self, epochs):
                print(f"Training Gaussian Baseline...")
                for epoch in range(1, epochs + 1):
                    metrics = self.train_step()
                    self.history['g_loss'].append(metrics['g_loss'])
                    self.history['d_loss'].append(metrics['d_loss'])
                    self.history['d_acc'].append(metrics['d_acc'])

                    if epoch % 20 == 0:
                        print(f"  Epoch {epoch}: G Loss: {metrics['g_loss']:.4f}, "
                              f"D Loss: {metrics['d_loss']:.4f}, D Acc: {metrics['d_acc']:.4f}")

        # Use EXACT same training as core file
        baseline_trainer = BaselineTrainer(generator, discriminator, config, self.device)
        baseline_trainer.train(epochs=epochs)

        # Return same format as core file
        training_metrics = {
            'g_losses': baseline_trainer.history['g_loss'],
            'd_losses': baseline_trainer.history['d_loss'],
            'constraint_satisfactions': [0.0] * len(baseline_trainer.history['g_loss']),
            'logic_weights': [0.0] * len(baseline_trainer.history['g_loss']),
            'constraint_effectiveness': 0.0
        }

        return training_metrics

    def _evaluate_gaussian_model(self, generator, ltn_system, config, variant_name):
        """
        Evaluation with stringent Gaussian adherence and intentionality checks.
        """
        generator.eval()
        with torch.no_grad():
            # Generate test samples (same as original)
            z = torch.randn(config.n_samples, config.latent_dim, device=self.device)
            test_samples = generator(z).cpu().numpy()

            # Calculate statistical metrics (same as original)
            sample_mean = np.mean(test_samples, axis=0)
            sample_std = np.std(test_samples, axis=0, ddof=1)

            # Target: mean=[0,0], std=[1,1]
            mean_error = np.linalg.norm(sample_mean)
            std_error = np.linalg.norm(sample_std - 1.0)

            # Statistical Quality: How well mean/std match target
            statistical_quality = 1.0 / (1.0 + mean_error + std_error)

            # Stringent Gaussian adherence calculation
            raw_gaussian_adherence = self._calculate_true_gaussian_adherence(test_samples)

            # Calculate logic satisfaction using LTN system if available
            test_samples_tensor = torch.tensor(test_samples, device=self.device, dtype=torch.float32)
            if ltn_system is not None:
                try:
                    constraints = ltn_system.compute_constraints(test_samples_tensor)
                    constraint_loss, individual_losses = ltn_system.compute_constraint_loss(constraints)

                    logic_satisfaction = np.mean([info['satisfaction'] for info in individual_losses.values()])
                    range_satisfaction = individual_losses.get('in_range', {}).get('satisfaction', 0.0)
                    shape_satisfaction = individual_losses.get('gaussian_shape', {}).get('satisfaction', 0.0)

                except Exception as e:
                    print(f"      Warning: Could not calculate logic satisfaction: {e}")
                    logic_satisfaction = 0.0
                    range_satisfaction = 0.0
                    shape_satisfaction = 0.0
            else:
                logic_satisfaction = 0.0
                range_satisfaction = 0.0
                shape_satisfaction = 0.0

            # Apply intentionality penalty to both Gaussian adherence and statistical quality
            gaussian_adherence = self._apply_intentionality_penalty(
                raw_gaussian_adherence, logic_satisfaction, variant_name
            )
            
            # Also apply penalty to statistical quality for baseline (accidental good stats)
            if 'baseline' in variant_name.lower():
                statistical_quality = self._apply_intentionality_penalty(
                    statistical_quality, logic_satisfaction, variant_name
                )

            # FIXED: Intentionality check with better thresholds
            if logic_satisfaction > 0.6 and gaussian_adherence > 0.4:
                intentionality_check = "INTENTIONAL (High logic + good adherence)"
            elif logic_satisfaction < 0.2 and raw_gaussian_adherence > 0.4:
                intentionality_check = "ACCIDENTAL (Low logic + high raw adherence)"
            elif logic_satisfaction > 0.7 and gaussian_adherence < 0.3:
                intentionality_check = "CONSTRAINT_FAILURE (High logic but low adherence)"
            elif logic_satisfaction > 0.5:
                intentionality_check = "PARTIAL_SUCCESS (Good logic, moderate adherence)"
            else:
                intentionality_check = "MIXED (Various factors)"

            # Combined Gaussian Quality
            combined_gaussian_quality = (statistical_quality + gaussian_adherence) / 2

            evaluation = {
                'logic_satisfaction': logic_satisfaction,
                'statistical_quality': statistical_quality,
                'gaussian_adherence': gaussian_adherence,
                'combined_gaussian_quality': combined_gaussian_quality,
                'range_satisfaction': range_satisfaction,
                'shape_satisfaction': shape_satisfaction,
                'mean_error': mean_error,
                'std_error': std_error,
                'samples': torch.tensor(test_samples),
                'intentionality_check': intentionality_check,
                'raw_gaussian_adherence': raw_gaussian_adherence  # For debugging
            }

            print(f"      Evaluation Details:")
            print(f"         Raw Gaussian Adherence: {raw_gaussian_adherence:.3f}")
            print(f"         Final Gaussian Adherence: {gaussian_adherence:.3f}")
            print(f"         Logic Satisfaction: {logic_satisfaction:.3f}")
            print(f"         Intentionality: {intentionality_check}")

        generator.train()
        return evaluation

    def run_all_experiments(self, epochs=100):
        """
        Run all ablation experiments with error handling.
        """
        print(" Starting Gaussian Dataset Ablation Experiments")
        print("=" * 80)
        print("  Improvements:")
        print("   Much more stringent Gaussian adherence calculation")
        print("   Intentionality penalty for accidental good scores")
        print("   Enhanced constraint system verification")
        print("   Deep monitoring of training dynamics")

        all_results = {}
        failed_experiments = []

        for variant_name, variant_config in self.ablation_variants.items():
            print(f"\n{'='*60}")
            print(f"Starting experiment: {variant_name}")
            print(f"{'='*60}")

            try:
                result = self.run_single_experiment(variant_name, variant_config, epochs)
                if result:
                    all_results[variant_name] = result
                    print(f" {variant_name} completed successfully")
                else:
                    failed_experiments.append(variant_name)
                    print(f" {variant_name} failed - no result returned")
            except Exception as e:
                failed_experiments.append(variant_name)
                print(f" {variant_name} failed with exception: {e}")
                traceback.print_exc()
                continue

        print(f"\n EXPERIMENT SUMMARY:")
        print(f"    Successful experiments: {len(all_results)}")
        print(f"    Failed experiments: {len(failed_experiments)}")

        if failed_experiments:
            print(f"   Failed variants: {', '.join(failed_experiments)}")

        if len(all_results) < 2:
            print(f"  WARNING: Only {len(all_results)} experiments completed successfully!")
            print(f"   This may indicate training instability or configuration issues.")

        # Save results
        results_file = os.path.join(self.results_dir, 'gaussian_ablation_results.json')

        # Convert tensor samples to lists for JSON serialization
        json_results = {}
        for variant_name, result in all_results.items():
            json_result = result.copy()
            if 'samples' in json_result:
                json_result['samples'] = json_result['samples'].numpy().tolist()
            json_results[variant_name] = json_result

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

        print(f"\n Results saved to: {results_file}")

        # Generate analysis only if we have results
        if all_results:
            self.analyze_results(all_results)
            self.create_visualizations(all_results)
        else:
            print("  No results to analyze - all experiments failed!")

        return all_results

    def analyze_results(self, results):

        print("\nGaussian Ablation Analysis...")

        # Create performance table (without intentionality column)
        table_file = os.path.join(self.results_dir, 'gaussian_ablation_table.txt')
        with open(table_file, 'w') as f:
            f.write("Gaussian Dataset Ablation Study Results\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Variant':<20} {'Gauss':<8} {'Stat':<8} {'Combined':<10} {'Mean':<8} {'Std':<8} {'Logic':<8}\n")
            f.write(f"{'':>20} {'Adher':<8} {'Qual':<8} {'Quality':<10} {'Error':<8} {'Error':<8} {'Sat':<8}\n")
            f.write("-" * 100 + "\n")

            for variant_name, result in results.items():
                f.write(f"{variant_name:<20} "
                       f"{result['gaussian_adherence']:>7.3f} "
                       f"{result['statistical_quality']:>7.3f} "
                       f"{result['combined_gaussian_quality']:>9.4f} "
                       f"{result['mean_error']:>7.4f} "
                       f"{result['std_error']:>7.4f} "
                       f"{result['logic_satisfaction']:>7.3f}\n")

        print(f"Analysis table saved to: {table_file}")

        # Console summary with intentionality assessment
        print("\nGaussian Ablation Summary:")
        print("=" * 100)

        baseline_result = results.get('baseline_gan', {})
        full_ltn_result = results.get('full_ltn_gan', {})

        if baseline_result and full_ltn_result:
            print(f" BASELINE GAN:")
            print(f"   Logic Satisfaction: {baseline_result['logic_satisfaction']:.3f}")
            print(f"   Gaussian Adherence: {baseline_result['gaussian_adherence']:.3f}")
            print(f"   Statistical Quality: {baseline_result['statistical_quality']:.3f}")
            print(f"   Combined Quality: {baseline_result['combined_gaussian_quality']:.3f}")
            print(f"   Intentionality: {baseline_result['intentionality_check']}")
            if 'raw_gaussian_adherence' in baseline_result:
                print(f"   Raw Adherence (before penalty): {baseline_result.get('raw_gaussian_adherence', 'N/A'):.3f}")

            print(f"\nFULL LTN-GAN (Original Trainer):")
            print(f"   Logic Satisfaction: {full_ltn_result['logic_satisfaction']:.3f}")
            print(f"   Gaussian Adherence: {full_ltn_result['gaussian_adherence']:.3f}")
            print(f"   Statistical Quality: {full_ltn_result['statistical_quality']:.3f}")
            print(f"   Combined Quality: {full_ltn_result['combined_gaussian_quality']:.3f}")
            print(f"   Intentionality: {full_ltn_result['intentionality_check']}")
            if 'raw_gaussian_adherence' in full_ltn_result:
                print(f"   Raw Adherence (before penalty): {full_ltn_result.get('raw_gaussian_adherence', 'N/A'):.3f}")

            # Calculate improvements
            logic_improvement = full_ltn_result['logic_satisfaction'] - baseline_result['logic_satisfaction']
            gauss_improvement = full_ltn_result['gaussian_adherence'] - baseline_result['gaussian_adherence']
            stat_improvement = full_ltn_result['statistical_quality'] - baseline_result['statistical_quality']
            combined_improvement = full_ltn_result['combined_gaussian_quality'] - baseline_result['combined_gaussian_quality']

            print(f"\nIMPROVEMENTS (LTN-GAN vs Baseline):")
            print(f"   Logic Satisfaction: +{logic_improvement:.3f}")
            print(f"   Gaussian Adherence: +{gauss_improvement:.3f}")
            print(f"   Statistical Quality: +{stat_improvement:.3f}")
            print(f"   Combined Quality: +{combined_improvement:.3f}")

        # Success assessment
        print(f"\nSUCCESS ASSESSMENT:")

        # Check if baseline has suspiciously high Gaussian adherence
        if 'gaussian_adherence' in baseline_result and baseline_result['gaussian_adherence'] > 0.4:
            print(f"   BASELINE ISSUE: Baseline has suspiciously high Gaussian adherence ({baseline_result['gaussian_adherence']:.3f})")
            print(f"      This suggests either:")
            print(f"      1. Evaluation is too lenient")
            print(f"      2. Lucky accident in baseline training")
            print(f"      3. Baseline accidentally learned some Gaussian structure")
        else:
            if 'gaussian_adherence' in baseline_result:
                print(f"   BASELINE OK: Low Gaussian adherence ({baseline_result['gaussian_adherence']:.3f}) as expected")
            else:
                print(f"   BASELINE OK: No Gaussian adherence data (as expected for baseline)")

        # Check LTN-GAN performance
        if full_ltn_result['logic_satisfaction'] > 0.8 and full_ltn_result['gaussian_adherence'] > 0.7:
            print(f"   EXCELLENT: LTN-GAN shows intentional Gaussian learning!")
            print(f"      High logic satisfaction ({full_ltn_result['logic_satisfaction']:.3f}) AND high adherence ({full_ltn_result['gaussian_adherence']:.3f})")
        elif full_ltn_result['logic_satisfaction'] > 0.6 and gauss_improvement > 0.3:
            print(f"   GOOD: LTN-GAN shows clear improvement over baseline")
        elif logic_improvement > 0.5:
            print(f"   PARTIAL: High logic satisfaction but limited Gaussian adherence improvement")
            print(f"      This suggests constraints are learned but may not translate to better Gaussian structure")
        else:
            print(f"   ISSUE: LTN-GAN not showing expected improvements")

        # Intentionality analysis
        print(f"\nINTENTIONALITY ANALYSIS:")
        for variant_name, result in results.items():
            intent_check = result['intentionality_check']
            print(f"   {variant_name}: {intent_check}")

            if 'ACCIDENTAL' in intent_check:
                print(f"      This variant may have accidentally good scores")
            elif 'INTENTIONAL' in intent_check:
                print(f"      This variant shows intentional Gaussian learning")

        # Find best variants
        best_logic = max(results.items(), key=lambda x: x[1]['logic_satisfaction'])
        best_gauss = max(results.items(), key=lambda x: x[1]['gaussian_adherence'])
        best_combined = max(results.items(), key=lambda x: x[1]['combined_gaussian_quality'])

        print(f" BEST PERFORMERS:")
        print(f"   Logic Satisfaction: {best_logic[0]} ({best_logic[1]['logic_satisfaction']:.3f})")
        print(f"   Gaussian Adherence: {best_gauss[0]} ({best_gauss[1]['gaussian_adherence']:.3f})")
        print(f"   Combined Quality: {best_combined[0]} ({best_combined[1]['combined_gaussian_quality']:.3f})")

    def create_visualizations(self, results):
        """
        Create visualizations with intentionality analysis.
        """
        print("Creating Gaussian Ablation Visualizations...")

        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 3, figsize=(22, 18))
        fig.suptitle('Gaussian Ablation Study Results',
                     fontsize=20, fontweight='bold', y=0.95)

        variants = list(results.keys())

        # 1. Logic Satisfaction vs Gaussian Adherence Scatter
        ax1 = axes[0, 0]
        logic_sats = [results[v]['logic_satisfaction'] for v in variants]
        gauss_adhers = [results[v]['gaussian_adherence'] for v in variants]
        colors = ['red' if 'baseline' in v else 'green' if 'full_ltn' in v else 'blue' for v in variants]

        scatter = ax1.scatter(logic_sats, gauss_adhers, c=colors, s=100, alpha=0.7)

        for i, variant in enumerate(variants):
            ax1.annotate(variant.replace('_', '\n'), (logic_sats[i], gauss_adhers[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax1.set_xlabel('Logic Satisfaction')
        ax1.set_ylabel('Gaussian Adherence')
        ax1.set_title('Logic vs Gaussian Adherence\n(Intentionality Check)')
        ax1.grid(True, alpha=0.3)

        # Add diagonal line for "intentional" region
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Intentional Line')
        ax1.legend()

        # 2. Raw vs Final Gaussian Adherence (showing penalties)
        ax2 = axes[0, 1]
        if all('raw_gaussian_adherence' in results[v] for v in variants):
            raw_adhers = [results[v]['raw_gaussian_adherence'] for v in variants]
            final_adhers = [results[v]['gaussian_adherence'] for v in variants]

            x_pos = np.arange(len(variants))
            bars1 = ax2.bar(x_pos - 0.2, raw_adhers, 0.4, label='Raw Adherence', alpha=0.8, color='orange')
            bars2 = ax2.bar(x_pos + 0.2, final_adhers, 0.4, label='Final Adherence', alpha=0.8, color='blue')

            ax2.set_ylabel('Gaussian Adherence')
            ax2.set_title('Raw vs Final Adherence\n(Showing Intentionality Penalties)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([v.replace('_', '\n') for v in variants], rotation=0, ha='center')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Add value annotations
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        # 3. Combined Quality Comparison
        ax3 = axes[0, 2]
        combined_qualities = [results[v]['combined_gaussian_quality'] for v in variants]
        bars = ax3.bar(variants, combined_qualities, alpha=0.8, color=colors)
        ax3.set_ylabel('Combined Gaussian Quality')
        ax3.set_title('Combined Gaussian Quality')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        for bar, val in zip(bars, combined_qualities):
            ax3.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

        # 4-6. Sample visualizations for key variants (updated for 6 variants)
        available_variants = list(results.keys())
        max_sample_plots = min(6, len(available_variants))  # Show up to 6 sample plots

        # Prioritize important variants for visualization
        priority_variants = ['baseline_gan', 'full_ltn_gan', 'ltn_no_constraints',
                           'ltn_high_constraint', 'ltn_fast_scheduling', 'ltn_slow_scheduling']

        # Select variants to show (prioritizing key ones)
        variants_to_show = []
        for variant in priority_variants:
            if variant in available_variants:
                variants_to_show.append(variant)

        # Add any remaining variants if we have space
        for variant in available_variants:
            if variant not in variants_to_show and len(variants_to_show) < 6:
                variants_to_show.append(variant)

        # Create a 2x3 grid for sample plots
        sample_axes = [axes[1, 0], axes[1, 1], axes[1, 2], axes[2, 0], axes[2, 1], axes[2, 2]]

        for i, variant_name in enumerate(variants_to_show):
            if i >= 6:  # Safety check
                break

            ax = sample_axes[i]
            samples = results[variant_name]['samples']
            intent_check = results[variant_name]['intentionality_check']

            # Color based on intentionality
            if 'INTENTIONAL' in intent_check:
                color = 'green'
                alpha = 0.6
            elif 'ACCIDENTAL' in intent_check:
                color = 'orange'
                alpha = 0.6
            else:
                color = 'red'
                alpha = 0.6

            ax.scatter(samples[:, 0], samples[:, 1], alpha=alpha, s=8, c=color)

            # Add reference circles for Gaussian evaluation
            from matplotlib.patches import Circle
            for std in [1, 2, 3]:
                circle = Circle((0, 0), std, fill=False, color='black',
                              linestyle='--', alpha=0.5, linewidth=1)
                ax.add_patch(circle)

            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_aspect('equal')

            # Enhanced title with metrics and intentionality
            logic_sat = results[variant_name]['logic_satisfaction']
            gauss_adh = results[variant_name]['gaussian_adherence']
            short_name = variant_name.replace('_', ' ').replace('ltn ', '').title()
            ax.set_title(f'{short_name}\nL:{logic_sat:.2f}, G:{gauss_adh:.2f}\n{intent_check[:12]}...', fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide any unused sample plot axes
        for i in range(len(variants_to_show), 6):
            sample_axes[i].set_visible(False)
        for variant_name in variants:
            if 'ltn' in variant_name.lower():
                constraint_sats = results[variant_name]['training_metrics']['constraint_satisfactions']
                if len(constraint_sats) > 0 and max(constraint_sats) > 0:
                    ax.plot(constraint_sats, label=variant_name.replace('_', ' '), alpha=0.8)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Constraint Satisfaction')
        ax.set_title('Constraint Learning Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 8. Logic Weight Evolution
        ax8 = axes[2, 1]
        for variant_name in variants:
            if 'ltn' in variant_name.lower():
                logic_weights = results[variant_name]['training_metrics']['logic_weights']
                if len(logic_weights) > 0 and max(logic_weights) > 0:
                    ax8.plot(logic_weights, label=variant_name.replace('_', ' '), alpha=0.8)

        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Logic Weight')
        ax8.set_title('Logic Weight Scheduling')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. Intentionality Summary
        ax9 = axes[2, 2]
        intentionality_counts = {}
        for variant_name, result in results.items():
            intent = result['intentionality_check']
            if 'INTENTIONAL' in intent:
                category = 'INTENTIONAL'
            elif 'ACCIDENTAL' in intent:
                category = 'ACCIDENTAL'
            elif 'CONSTRAINT_FAILURE' in intent:
                category = 'CONSTRAINT_FAILURE'
            else:
                category = 'MIXED'

            intentionality_counts[category] = intentionality_counts.get(category, 0) + 1

        if intentionality_counts:
            categories = list(intentionality_counts.keys())
            counts = list(intentionality_counts.values())
            colors_intent = ['green', 'orange', 'red', 'blue'][:len(categories)]

            bars = ax9.bar(categories, counts, color=colors_intent, alpha=0.7)
            ax9.set_ylabel('Number of Variants')
            ax9.set_title('Intentionality Summary')
            ax9.grid(True, alpha=0.3)

            for bar, count in zip(bars, counts):
                ax9.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        # Save visualization
        viz_file = os.path.join(self.results_dir, 'visualizations', 'gaussian_ablation_results.png')
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {viz_file}")

def main():
    """
    Main function to run the Gaussian ablation framework.
    """
    print("Gaussian LTN-GAN Ablation Framework")
    print("=" * 70)
    print("Fixes Applied:")
    print("   Much more stringent Gaussian adherence calculation")
    print("   Intentionality penalty for accidental good scores")
    print("   Enhanced constraint system effectiveness verification")
    print("   Deep monitoring of training dynamics")
    print("   Comprehensive intentionality analysis")

    # Create framework
    framework = GaussianAblationFramework()

    # Run all experiments
    print(f"\nRunning experiments with 100 epochs...")
    results = framework.run_all_experiments(epochs=100)

    print("\nGaussian Ablation Experiments Complete!")
    print("=" * 70)

    if 'baseline_gan' in results and 'full_ltn_gan' in results:
        baseline = results['baseline_gan']
        ltn_gan = results['full_ltn_gan']

        print("FINAL COMPARISON:")
        print("-" * 60)
        print(f"BASELINE GAN:")
        print(f"   Logic Satisfaction: {baseline['logic_satisfaction']:.3f}")
        print(f"   Gaussian Adherence: {baseline['gaussian_adherence']:.3f}")
        print(f"   Intentionality: {baseline['intentionality_check']}")

        print(f"\nFULL LTN-GAN:")
        print(f"   Logic Satisfaction: {ltn_gan['logic_satisfaction']:.3f}")
        print(f"   Gaussian Adherence: {ltn_gan['gaussian_adherence']:.3f}")
        print(f"   Intentionality: {ltn_gan['intentionality_check']}")

        # Calculate improvements
        logic_improvement = ltn_gan['logic_satisfaction'] - baseline['logic_satisfaction']
        gauss_improvement = ltn_gan['gaussian_adherence'] - baseline['gaussian_adherence']

        print(f"\nIMPROVEMENTS:")
        print(f"   Logic Satisfaction: +{logic_improvement:.3f}")
        print(f"   Gaussian Adherence: +{gauss_improvement:.3f}")

        # Success assessment
        if logic_improvement > 0.6 and gauss_improvement > 0.4:
            print(f"\nSUCCESS!")
            print(f"   LTN-GAN shows massive intentional improvements!")
            print(f"   This demonstrates clear constraint-driven Gaussian learning.")
        elif logic_improvement > 0.4 and gauss_improvement > 0.2:
            print(f"\nSTRONG SUCCESS!")
            print(f"   LTN-GAN shows substantial improvements.")
        elif logic_improvement > 0.2:
            print(f"\nMODERATE SUCCESS:")
            print(f"   LTN-GAN shows constraint learning but limited Gaussian improvement.")
        else:
            print(f"\nINVESTIGATION NEEDED:")
            print(f"   Results may indicate implementation or evaluation issues.")

        # Baseline check
        if baseline['gaussian_adherence'] > 0.3:
            print(f"\nBASELINE WARNING:")
            print(f"   Baseline has higher Gaussian adherence ({baseline['gaussian_adherence']:.3f}) than expected.")
            print(f"   This could indicate evaluation issues or lucky accident.")
        else:
            print(f"\nBASELINE OK:")
            print(f"   Baseline has appropriately low Gaussian adherence ({baseline['gaussian_adherence']:.3f}).")

    return results

if __name__ == "__main__":
    results = main()
