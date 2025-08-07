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

# Import Grid LTN-GAN implementation
try:
    from grid_ltn_gan import (
        SharedConfig,
        GridGenerator,
        GridDiscriminator,
        SimpleLTNSystem,
        BaselineGridTrainer,
        LTNGridTrainer
    )
    import ltn
except ImportError as e:
    print(f"Error importing Grid LTN-GAN modules: {e}")
    sys.exit(1)

class GridAblationFramework:
    """
    Ablation framework for Grid LTN-GAN experiments.
    Tests different components and configurations.
    """

    def __init__(self, results_dir="grid_ablation_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = SharedConfig()

        # Define ablation variants
        self.ablation_variants = {
            'baseline_gan': {
                'use_ltn_constraints': False,
                'use_ltn_trainer': False,
                'description': 'Baseline GAN without LTN constraints'
            },
            'full_ltn_gan': {
                'use_ltn_constraints': True,
                'use_ltn_trainer': True,
                'config_overrides': {},
                'description': 'Full LTN-GAN with grid constraints'
            },
            'ltn_no_constraints': {
                'use_ltn_constraints': True,
                'use_ltn_trainer': True,
                'config_overrides': {
                    'logic_weight_start': 0.0,
                    'logic_weight_end': 0.0
                },
                'description': 'LTN-GAN with disabled constraints'
            },
            'ltn_high_constraint': {
                'use_ltn_constraints': True,
                'use_ltn_trainer': True,
                'config_overrides': {
                    'logic_weight_start': 0.5,
                    'logic_weight_end': 2.0,
                    'epochs': 100
                },
                'description': 'LTN-GAN with high constraint weight'
            },
            'ltn_fast_scheduling': {
                'use_ltn_constraints': True,
                'use_ltn_trainer': True,
                'config_overrides': {
                    'logic_weight_start': 0.5,
                    'logic_weight_end': 1.5,
                    'ramp_epochs': 40
                },
                'description': 'LTN-GAN with fast scheduling'
            },
            'ltn_slow_scheduling': {
                'use_ltn_constraints': True,
                'use_ltn_trainer': True,
                'config_overrides': {
                    'logic_weight_start': 0.1,
                    'logic_weight_end': 1.0,
                    'ramp_epochs': 100
                },
                'description': 'LTN-GAN with slow scheduling'
            }
        }

        # Grid ablation framework initialized

    def run_single_experiment(self, variant_name, variant_config, epochs=100):
        """
        Run a single ablation experiment.
        """
        print(f"Running Grid Experiment: {variant_name}")

        try:
            if variant_config.get('use_ltn_trainer', False):
                if variant_config['use_ltn_constraints']:
                    # Create LTN trainer
                    trainer = LTNGridTrainer(latent_dim=100)

                    # Test constraint system effectiveness
                    print(f"       Testing constraint system effectiveness...")
                    test_samples = self._generate_test_samples()
                    effectiveness = self._test_constraint_effectiveness(trainer.ltn_system, test_samples)
                    print(f"         Constraint effectiveness: {effectiveness:.3f}")

                    print(f"      Using LTN-GAN trainer")

                    # Train with monitoring
                    self._train_with_monitoring(trainer, epochs, variant_config.get('config_overrides', {}))

                    # Get training metrics
                    training_metrics = {
                        'g_losses': trainer.losses['g_losses'],
                        'd_losses': trainer.losses['d_losses'],
                        'd_accuracies': trainer.losses['d_accuracies'],
                        'constraint_satisfactions': trainer.losses['constraint_satisfactions'],
                        'logic_weights': trainer.losses['logic_weights'],
                        'constraint_effectiveness': effectiveness
                    }

                    # Evaluate final performance
                    final_evaluation = self._evaluate_grid_model(trainer, variant_name)

                else:
                    raise ValueError("Cannot use LTN trainer without LTN system")

            else:
                # Simple baseline trainer (no LTN)
                print(f"      Training baseline GAN...")
                trainer = BaselineGridTrainer(latent_dim=100)
                trainer.train(epochs=epochs)

                training_metrics = {
                    'g_losses': trainer.losses['g_losses'],
                    'd_losses': trainer.losses['d_losses'],
                    'd_accuracies': trainer.losses['d_accuracies'],
                    'constraint_satisfactions': [0.0] * len(trainer.losses['g_losses']),
                    'logic_weights': [0.0] * len(trainer.losses['g_losses']),
                    'constraint_effectiveness': 0.0
                }

                final_evaluation = self._evaluate_grid_model(trainer, variant_name)

            # Combine results
            result = {
                'variant': variant_name,
                'grid_clustering_score': final_evaluation['grid_clustering_score'],
                'position_coverage': final_evaluation['position_coverage'],
                'clustering_quality': final_evaluation['clustering_quality'],
                'logic_satisfaction': final_evaluation['logic_satisfaction'],
                'overall_grid_performance': final_evaluation['overall_grid_performance'],
                'position_distributions': final_evaluation['position_distributions'],
                'samples_in_targets': final_evaluation['samples_in_targets'],
                'training_metrics': training_metrics,
                'samples': final_evaluation['samples']
            }

            print(f"    Grid Clustering Score: {final_evaluation['grid_clustering_score']:.3f}")
            print(f"    Position Coverage: {final_evaluation['position_coverage']:.3f}")
            print(f"    Clustering Quality: {final_evaluation['clustering_quality']:.3f}")
            print(f"    Samples in Targets: {final_evaluation['samples_in_targets']}")

            return result

        except Exception as e:
            print(f"    Error in {variant_name}: {e}")
            traceback.print_exc()
            return None

    def _generate_test_samples(self):
        """Generate test samples for constraint effectiveness testing."""
        # Perfect grid samples (should score high)
        perfect_samples = []
        for pos in self.config.grid_positions:
            noise = torch.randn(25, 2) * 0.005  # Very tight clusters
            perfect_samples.append(pos.unsqueeze(0).repeat(25, 1) + noise)
        perfect_samples = torch.cat(perfect_samples, dim=0)

        # Random samples (should score low)
        random_samples = torch.randn(100, 2) * 0.1

        return {'perfect': perfect_samples, 'random': random_samples}

    def _test_constraint_effectiveness(self, ltn_system, test_samples):
        """Test how well the constraint system distinguishes good vs bad samples."""
        try:
            # Test perfect grid samples
            perfect_constraints = ltn_system.compute_constraints(test_samples['perfect'])
            perfect_loss = ltn_system.compute_constraint_loss(perfect_constraints)
            perfect_satisfaction = 1.0 - perfect_loss.item()

            # Test random samples
            random_constraints = ltn_system.compute_constraints(test_samples['random'])
            random_loss = ltn_system.compute_constraint_loss(random_constraints)
            random_satisfaction = 1.0 - random_loss.item()

            # Effectiveness is the difference
            effectiveness = perfect_satisfaction - random_satisfaction

            print(f"         Perfect samples satisfaction: {perfect_satisfaction:.3f}")
            print(f"         Random samples satisfaction: {random_satisfaction:.3f}")

            return effectiveness
        except Exception as e:
            print(f"         Warning: Could not test constraint effectiveness: {e}")
            return 0.0

    def _train_with_monitoring(self, trainer, epochs, config_overrides):
        """Enhanced training with monitoring for Grid LTN-GAN."""
        print(f"       Training with constraint monitoring...")

        # Apply config overrides for logic weight scheduling
        logic_weight_start = config_overrides.get('logic_weight_start', 0.1)
        logic_weight_end = config_overrides.get('logic_weight_end', 0.5)
        ramp_epochs = config_overrides.get('ramp_epochs', epochs)

        print(f"         Logic weight: {logic_weight_start} → {logic_weight_end} over {ramp_epochs} epochs")

        for epoch in range(epochs):
            # Calculate logic weight with custom scheduling
            if ramp_epochs > 0 and epoch < ramp_epochs:
                progress = epoch / ramp_epochs
                logic_weight = logic_weight_start + progress * (logic_weight_end - logic_weight_start)
            else:
                logic_weight = logic_weight_end

            # Custom training step with monitoring
            metrics = trainer.train_step(logic_weight=logic_weight)

            # Store metrics
            trainer.losses['g_losses'].append(metrics['g_loss'])
            trainer.losses['d_losses'].append(metrics['d_loss'])
            trainer.losses['d_accuracies'].append(metrics['d_accuracy'])
            trainer.losses['constraint_losses'].append(metrics['constraint_loss'])
            trainer.losses['constraint_satisfactions'].append(metrics['logic_satisfaction'])
            trainer.losses['adversarial_losses'].append(metrics['adversarial_loss'])
            trainer.losses['weighted_constraint_losses'].append(metrics['weighted_constraint_loss'])
            trainer.losses['logic_weights'].append(metrics['logic_weight'])

            # Enhanced progress reporting
            if (epoch + 1) % 25 == 0:
                print(f"         Epoch {epoch+1}: G Loss: {metrics['g_loss']:.4f}, "
                      f"Constraint Sat: {metrics['logic_satisfaction']:.3f}, "
                      f"Logic Weight: {logic_weight:.3f}")

    def _evaluate_grid_model(self, trainer, variant_name):
        """
        Comprehensive evaluation of grid model performance.
        """
        trainer.generator.eval()
        with torch.no_grad():
            # Generate test samples
            test_samples = trainer.generate_samples(1000)

            # Calculate grid-specific metrics
            grid_metrics = self._calculate_grid_metrics(test_samples)

            # Calculate constraint satisfaction if LTN system exists
            if hasattr(trainer, 'ltn_system'):
                try:
                    test_samples_gpu = test_samples.to(self.device)
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

    def _calculate_grid_metrics(self, samples):
        """
        Calculate comprehensive grid-specific metrics.
        """
        # Analyze position distribution
        position_counts = [0, 0, 0, 0]
        samples_in_targets = 0
        target_radius = self.config.target_radius

        for sample in samples:
            distances = [torch.norm(sample - pos).item() for pos in self.config.grid_positions]
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
        for i, pos in enumerate(self.config.grid_positions):
            # Find samples assigned to this position
            position_samples = []
            for sample in samples:
                distances = [torch.norm(sample - p).item() for p in self.config.grid_positions]
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

    def run_all_experiments(self, epochs=100):
        """
        Run all ablation experiments for Grid dataset.
        """
        print("Starting Grid Dataset Ablation Experiments")
        print("=" * 70)

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
        print(f"   Failed experiments: {len(failed_experiments)}")

        if failed_experiments:
            print(f"   Failed variants: {', '.join(failed_experiments)}")

        # Save results
        results_file = os.path.join(self.results_dir, 'grid_ablation_results.json')

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

        # Generate analysis
        if all_results:
            self.analyze_results(all_results)
            self.create_visualizations(all_results)
        else:
            print("No results to analyze - all experiments failed!")

        return all_results

    def analyze_results(self, results):
        """
        Analyze and create tables from the ablation results.
        """
        print("\n Analyzing Grid Ablation Results...")

        # Create performance table
        table_file = os.path.join(self.results_dir, 'grid_ablation_table.txt')
        with open(table_file, 'w') as f:
            f.write("Grid Dataset Ablation Study Results\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Variant':<20} {'Grid':<8} {'Coverage':<9} {'Quality':<8} {'Radial':<8} {'Overall':<8} {'Samples':<8} {'Logic':<8}\n")
            f.write(f"{'':>20} {'Cluster':<8} {'Score':<9} {'Score':<8} {'Align':<8} {'Perf':<8} {'In Targets':<8} {'Sat':<8}\n")
            f.write("-" * 100 + "\n")

            for variant_name, result in results.items():
                f.write(f"{variant_name:<20} "
                       f"{result['grid_clustering_score']:>7.3f} "
                       f"{result['position_coverage']:>8.3f} "
                       f"{result['clustering_quality']:>7.3f} "
                       f"{result.get('radial_alignment_score', 0.0):>7.3f} "
                       f"{result['overall_grid_performance']:>7.3f} "
                       f"{result['samples_in_targets']:>7} "
                       f"{result['logic_satisfaction']:>7.3f}\n")

        print(f" Analysis table saved to: {table_file}")

        # Console summary
        print("\n Grid Ablation Summary:")
        print("-" * 80)

        baseline_result = results.get('baseline_gan', {})
        full_ltn_result = results.get('full_ltn_gan', {})

        if baseline_result and full_ltn_result:
            print(f" BASELINE GAN:")
            print(f"   Grid Clustering Score: {baseline_result['grid_clustering_score']:.3f}")
            print(f"   Position Coverage: {baseline_result['position_coverage']:.3f}")
            print(f"   Clustering Quality: {baseline_result['clustering_quality']:.3f}")
            print(f"   Radial Alignment: {baseline_result.get('radial_alignment_score', 0.0):.3f}")
            print(f"   Overall Performance: {baseline_result['overall_grid_performance']:.3f}")
            print(f"   Samples in Targets: {baseline_result['samples_in_targets']}")

            print(f" FULL LTN-GAN:")
            print(f"   Grid Clustering Score: {full_ltn_result['grid_clustering_score']:.3f}")
            print(f"   Position Coverage: {full_ltn_result['position_coverage']:.3f}")
            print(f"   Clustering Quality: {full_ltn_result['clustering_quality']:.3f}")
            print(f"   Radial Alignment: {full_ltn_result.get('radial_alignment_score', 0.0):.3f}")
            print(f"   Logic Satisfaction: {full_ltn_result['logic_satisfaction']:.3f}")
            print(f"   Overall Performance: {full_ltn_result['overall_grid_performance']:.3f}")
            print(f"   Samples in Targets: {full_ltn_result['samples_in_targets']}")

            # Calculate improvements
            clustering_improvement = full_ltn_result['grid_clustering_score'] - baseline_result['grid_clustering_score']
            coverage_improvement = full_ltn_result['position_coverage'] - baseline_result['position_coverage']
            quality_improvement = full_ltn_result['clustering_quality'] - baseline_result['clustering_quality']
            overall_improvement = full_ltn_result['overall_grid_performance'] - baseline_result['overall_grid_performance']

            print(f" IMPROVEMENTS (LTN-GAN vs Baseline):")
            print(f"   Grid Clustering Score: +{clustering_improvement:.3f}")
            print(f"   Position Coverage: +{coverage_improvement:.3f}")
            print(f"   Clustering Quality: +{quality_improvement:.3f}")
            print(f"   Overall Performance: +{overall_improvement:.3f}")

            # Success assessment
            if clustering_improvement > 0.3 and coverage_improvement > 0.2:
                print(f" EXCELLENT SUCCESS!")
                print(f"   LTN-GAN shows significant improvements in grid clustering!")
            elif clustering_improvement > 0.1 and coverage_improvement > 0.1:
                print(f" GOOD SUCCESS!")
                print(f"   LTN-GAN shows substantial improvements.")
            elif clustering_improvement > 0.05:
                print(f" MODERATE SUCCESS:")
                print(f"   LTN-GAN shows improvements, but could be better.")
            else:
                print(f" ISSUE:")
                print(f"   LTN-GAN improvements are modest. Further investigation needed.")

        # Find best variants
        best_clustering = max(results.items(), key=lambda x: x[1]['grid_clustering_score'])
        best_overall = max(results.items(), key=lambda x: x[1]['overall_grid_performance'])

        print(f" BEST PERFORMERS:")
        print(f"   Grid Clustering: {best_clustering[0]} ({best_clustering[1]['grid_clustering_score']:.3f})")
        print(f"   Overall Performance: {best_overall[0]} ({best_overall[1]['overall_grid_performance']:.3f})")

    def create_visualizations(self, results):
        """
        Create comprehensive visualizations of the ablation results.
        """
        print(" Creating Grid Ablation Visualizations...")

        try:
            # Create comprehensive visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Grid Ablation Study Results', fontsize=16, fontweight='bold')

            variants = list(results.keys())

            # 1. Grid Clustering Score Comparison
            ax = axes[0, 0]
            clustering_scores = [results[v]['grid_clustering_score'] for v in variants]
            colors = ['red' if 'baseline' in v else 'green' if 'full_ltn' in v else 'blue' for v in variants]
            bars = ax.bar(variants, clustering_scores, alpha=0.8, color=colors)
            ax.set_ylabel('Grid Clustering Score')
            ax.set_title('Grid Clustering Performance')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            for bar, val in zip(bars, clustering_scores):
                ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

            # 2. Position Coverage Comparison
            ax = axes[0, 1]
            coverage_scores = [results[v]['position_coverage'] for v in variants]
            bars = ax.bar(variants, coverage_scores, alpha=0.8, color=colors)
            ax.set_ylabel('Position Coverage')
            ax.set_title('Position Coverage Quality')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            for bar, val in zip(bars, coverage_scores):
                ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

            # 3. Overall Performance Comparison
            ax = axes[0, 2]
            overall_scores = [results[v]['overall_grid_performance'] for v in variants]
            bars = ax.bar(variants, overall_scores, alpha=0.8, color=colors)
            ax.set_ylabel('Overall Performance')
            ax.set_title('Overall Grid Performance')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            for bar, val in zip(bars, overall_scores):
                ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

            # 4-6. Sample visualizations for key variants
            key_variants = ['baseline_gan', 'full_ltn_gan', 'ltn_high_constraint']
            for i, variant_name in enumerate(key_variants):
                if variant_name in results and i < 3:
                    ax = axes[1, i]
                    samples = results[variant_name]['samples']

                    # Color based on performance
                    if variant_name == 'baseline_gan':
                        color = 'red'
                        alpha = 0.6
                    elif variant_name == 'full_ltn_gan':
                        color = 'green'
                        alpha = 0.6
                    else:
                        color = 'blue'
                        alpha = 0.6

                    ax.scatter(samples[:, 0], samples[:, 1], alpha=alpha, s=8, c=color)

                    # Add grid positions and target circles
                    for j, pos in enumerate(self.config.grid_positions):
                        ax.scatter(pos[0], pos[1], c='red', s=80, marker='x', linewidths=3)
                        circle = plt.Circle((pos[0], pos[1]), self.config.target_radius,
                                          fill=False, color='red', linestyle='--', alpha=0.8)
                        ax.add_patch(circle)

                    ax.set_xlim(-0.08, 0.08)
                    ax.set_ylim(-0.08, 0.08)
                    ax.set_aspect('equal')

                    # Title with metrics
                    clustering_score = results[variant_name]['grid_clustering_score']
                    samples_in_targets = results[variant_name]['samples_in_targets']
                    short_name = variant_name.replace('_', ' ').title()
                    ax.set_title(f'{short_name}\nClustering: {clustering_score:.3f}\nIn Targets: {samples_in_targets}')
                    ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save visualization
            viz_file = os.path.join(self.results_dir, 'visualizations', 'grid_ablation_results.png')
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Grid ablation visualization saved to: {viz_file}")

        except Exception as e:
            print(f" Error creating visualizations: {e}")
            print("Skipping visualizations...")

def main():
    """
    Main function to run the Grid ablation framework.
    """
    print("Grid LTN-GAN Ablation Framework")
    print("=" * 40)

    # Create framework
    framework = GridAblationFramework()

    # Run all experiments
    results = framework.run_all_experiments(epochs=100)

    print("Grid ablation experiments complete!")

    if 'baseline_gan' in results and 'full_ltn_gan' in results:
        baseline = results['baseline_gan']
        ltn_gan = results['full_ltn_gan']

        print("Final Comparison Results:")
        print("-" * 40)
        print(f"Baseline GAN:")
        print(f"  Grid Clustering Score: {baseline['grid_clustering_score']:.3f}")
        print(f"  Position Coverage: {baseline['position_coverage']:.3f}")
        print(f"  Samples in Targets: {baseline['samples_in_targets']}")

        print(f"Full LTN-GAN:")
        print(f"  Grid Clustering Score: {ltn_gan['grid_clustering_score']:.3f}")
        print(f"  Position Coverage: {ltn_gan['position_coverage']:.3f}")
        print(f"  Logic Satisfaction: {ltn_gan['logic_satisfaction']:.3f}")
        print(f"  Samples in Targets: {ltn_gan['samples_in_targets']}")

        # Calculate improvements
        clustering_improvement = ltn_gan['grid_clustering_score'] - baseline['grid_clustering_score']
        coverage_improvement = ltn_gan['position_coverage'] - baseline['position_coverage']

        print(f"Improvements (LTN-GAN vs Baseline):")
        print(f"  Grid Clustering Score: +{clustering_improvement:.3f}")
        print(f"  Position Coverage: +{coverage_improvement:.3f}")

    return results

if __name__ == "__main__":
    results = main()
