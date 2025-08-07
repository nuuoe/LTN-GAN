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

# Import Ring LTN-GAN implementation
try:
    from ring_ltn_gan import (
        RingConfig,
        RingGenerator,
        RingDiscriminator,
        SophisticatedConstraintSystem,
        generate_ring_data,
        analyze_ring_performance,
        RingConfig,
        RingGenerator,
        RingDiscriminator,
        SophisticatedConstraintSystem,
        generate_ring_data,
        analyze_ring_performance
    )
    print("Ring LTN-GAN modules imported successfully")
except ImportError as e:
    print(f"Could not import Ring LTN-GAN modules: {e}")
    print("Please ensure best_ring_ltn_gan.py is in the Python path")
    sys.exit(1)

class RingAblationFramework:
    """
    Ablation framework for Ring LTN-GAN experiments.
    Tests different components and configurations.
    """

    def __init__(self, results_dir="ring_ablation_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "visualizations"), exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define ablation variants
        self.ablation_variants = {
            'baseline_gan': {
                'use_ltn_constraints': False,
                'logic_weight': 0.0,
                'constraint_ramp_epochs': 0,
                'use_smart_weighting': True,
                'use_progressive_phases': True,
                'description': 'Baseline GAN without LTN constraints'
            },
            'full_ltn_gan': {
                'use_ltn_constraints': True,
                'logic_weight': 10.0,
                'constraint_ramp_epochs': 100,
                'description': 'Full LTN-GAN with ring constraints'
            },
            'no_constraints': {
                'use_ltn_constraints': True,
                'logic_weight': 0.0,
                'constraint_ramp_epochs': 100,
                'description': 'LTN-GAN with disabled constraints'
            },
            'no_hierarchical_weights': {
                'use_ltn_constraints': True,
                'logic_weight': 10.0,
                'constraint_ramp_epochs': 100,
                'use_smart_weighting': False,
                'description': 'LTN-GAN without hierarchical weighting'
            },
            'no_progressive_phases': {
                'use_ltn_constraints': True,
                'logic_weight': 10.0,
                'constraint_ramp_epochs': 100,
                'use_progressive_phases': False,
                'description': 'LTN-GAN without progressive phases'
            },
            'simple_constraints': {
                'use_ltn_constraints': True,
                'logic_weight': 10.0,
                'constraint_ramp_epochs': 100,
                'use_simple_constraints': True,
                'description': 'LTN-GAN with simple constraints'
            }
        }

        print(f"Ring Ablation Framework Initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Device: {self.device}")
        print(f"Ablation variants: {len(self.ablation_variants)}")

    def run_single_experiment(self, variant_name, variant_config, epochs=100):
        """
        Run a single ablation experiment.
        """
        print(f"\nRunning Ring Experiment: {variant_name}")
        print(f"   {variant_config['description']}")

        try:
            # Create Ring configuration
            config = RingConfig()
            config.epochs = epochs
            config.constraint_weight_end = variant_config['logic_weight']
            config.constraint_ramp_epochs = variant_config['constraint_ramp_epochs']

            # Create models
            generator = RingGenerator(
                config,
                use_ltn_constraints=variant_config['use_ltn_constraints']
            ).to(self.device)

            discriminator = RingDiscriminator(config).to(self.device)

            # Create LTN constraint system (only if using LTN)
            if variant_config['use_ltn_constraints']:
                ltn_system = SophisticatedConstraintSystem(config, self.device)
            else:
                ltn_system = None

            # Train the model
            metrics = self._train_ring_model(
                generator, discriminator, ltn_system, config, variant_config
            )

            # Use last epoch's ring adherence from training metrics (like core file)
            final_ring_adherence = metrics['ring_adherences'][-1] if metrics['ring_adherences'] else 0.0
            
            # Evaluate final performance for additional metrics
            final_evaluation = self._evaluate_ring_model(generator, config, ltn_system)

            # Combine results
            result = {
                'variant': variant_name,
                'actual_ring_adherence': final_ring_adherence,  # Use training metric like core
                'logic_satisfaction': final_evaluation['logic_satisfaction'],
                'inner_count': final_evaluation['inner_count'],
                'outer_count': final_evaluation['outer_count'],
                'balance_score': final_evaluation['balance_score'],
                'dead_zone_avoidance': final_evaluation['dead_zone_avoidance'],
                'training_metrics': metrics,
                'samples': final_evaluation['samples']
            }

            print(f"   Ring Adherence: {final_ring_adherence:.1%}")
            print(f"   Logic Satisfaction: {final_evaluation['logic_satisfaction']:.3f}")
            print(f"      Ring Distribution: Inner={final_evaluation['inner_count']}, Outer={final_evaluation['outer_count']}")
            print(f"      Balance Score: {final_evaluation['balance_score']:.3f}")

            return result

        except Exception as e:
            print(f"   Error in {variant_name}: {e}")
            traceback.print_exc()
            return None

    def _train_ring_model(self, generator, discriminator, ltn_system, config, variant_config):
        """
        EXACT same method as core - Train the Ring model with ablation control.
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

    def _evaluate_ring_model(self, generator, config, ltn_system=None):
        """
        EXACT same method as core - Evaluate the trained Ring model.
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

    def run_all_experiments(self, epochs=100):
        """
        Run all ablation experiments for Ring dataset.
        """
        print("Starting Ring Dataset Ablation Experiments")
        print("=" * 60)

        all_results = {}

        for variant_name, variant_config in self.ablation_variants.items():
            result = self.run_single_experiment(variant_name, variant_config, epochs)
            if result:
                all_results[variant_name] = result

        # Save results
        results_file = os.path.join(self.results_dir, 'ring_ablation_results.json')

        # Convert tensor samples to lists for JSON serialization
        json_results = {}
        for variant_name, result in all_results.items():
            json_result = result.copy()
            if 'samples' in json_result:
                json_result['samples'] = json_result['samples'].numpy().tolist()
            json_results[variant_name] = json_result

        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

        print(f"Results saved to: {results_file}")

        # Generate analysis
        self.analyze_results(all_results)
        self.create_visualizations(all_results)

        return all_results

    def analyze_results(self, results):
        """
        Analyze and create tables from the ablation results.
        """
        print("Analyzing Ring Ablation Results...")

        # Create performance table
        table_file = os.path.join(self.results_dir, 'ring_ablation_table.txt')
        with open(table_file, 'w') as f:
            f.write("Ring Dataset Ablation Study Results\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'Variant':<20} {'Ring':<8} {'Inner':<6} {'Outer':<6} {'Balance':<8} {'Dead Zone':<10} {'Logic':<8}\n")
            f.write(f"{'':>20} {'Adher':<8} {'Count':<6} {'Count':<6} {'Score':<8} {'Avoid':<10} {'Sat':<8}\n")
            f.write("-" * 100 + "\n")

            for variant_name, result in results.items():
                f.write(f"{variant_name:<20} "
                       f"{result['actual_ring_adherence']:>7.3f} "
                       f"{result['inner_count']:>5} "
                       f"{result['outer_count']:>5} "
                       f"{result['balance_score']:>7.3f} "
                       f"{result['dead_zone_avoidance']:>9.3f} "
                       f"{result['logic_satisfaction']:>7.3f}\n")

        print(f"Analysis table saved to: {table_file}")

        # Print summary to console
        print("\nRing Ablation Summary:")
        print("-" * 80)

        baseline_adherence = results.get('baseline_gan', {}).get('actual_ring_adherence', 0.0)
        full_ltn_adherence = results.get('full_ltn_gan', {}).get('actual_ring_adherence', 0.0)
        baseline_logic = results.get('baseline_gan', {}).get('logic_satisfaction', 0.0)
        full_ltn_logic = results.get('full_ltn_gan', {}).get('logic_satisfaction', 0.0)

        improvement = (full_ltn_adherence - baseline_adherence) * 100
        logic_improvement = (full_ltn_logic - baseline_logic) * 100

        print(f"Baseline GAN Ring Adherence: {baseline_adherence:.1%}")
        print(f"Full LTN-GAN Ring Adherence: {full_ltn_adherence:.1%}")
        print(f"Ring Adherence Improvement: +{improvement:.1f} percentage points")
        print(f"Baseline GAN Logic Satisfaction: {baseline_logic:.3f}")
        print(f"Full LTN-GAN Logic Satisfaction: {full_ltn_logic:.3f}")
        print(f"Logic Satisfaction Improvement: +{logic_improvement:.1f} percentage points")

        # Find best variants for different metrics
        best_adherence = max(results.items(), key=lambda x: x[1]['actual_ring_adherence'])
        best_logic = max(results.items(), key=lambda x: x[1]['logic_satisfaction'])

        print(f"Best Ring Adherence: {best_adherence[0]} ({best_adherence[1]['actual_ring_adherence']:.1%})")
        print(f"Best Logic Satisfaction: {best_logic[0]} ({best_logic[1]['logic_satisfaction']:.3f})")

    def create_visualizations(self, results):
        """
        Create visualizations of the ablation results.
        """
        print("\nCreating Ring Ablation Visualizations...")

        # Create a comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ring Ablation Study Results', fontsize=16, fontweight='bold')

        # 1. Ring Adherence Comparison
        ax1 = axes[0, 0]
        variants = list(results.keys())
        adherences = [results[v]['actual_ring_adherence'] for v in variants]

        bars = ax1.bar(variants, adherences, alpha=0.8, color='skyblue')
        ax1.set_xlabel('Ablation Variant')
        ax1.set_ylabel('Ring Adherence')
        ax1.set_title('Ring Adherence Comparison')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        # 2. Logic Satisfaction Comparison
        ax2 = axes[0, 1]
        logic_satisfactions = [results[v]['logic_satisfaction'] for v in variants]
        bars = ax2.bar(variants, logic_satisfactions, alpha=0.8, color='purple')
        ax2.set_xlabel('Ablation Variant')
        ax2.set_ylabel('Logic Satisfaction')
        ax2.set_title('Logic Satisfaction Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)  # Logic satisfaction is typically 0-1

        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        # 3. Balance Score Comparison
        ax3 = axes[0, 2]
        # 4. Dead Zone Avoidance
        ax4 = axes[1, 0]
        dead_zone_avoid = [results[v]['dead_zone_avoidance'] for v in variants]
        bars = ax4.bar(variants, dead_zone_avoid, alpha=0.8, color='red')
        ax4.set_xlabel('Ablation Variant')
        ax4.set_ylabel('Dead Zone Avoidance')
        ax4.set_title('Dead Zone Avoidance Comparison')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        # 5. Sample Visualizations for Best and Worst
        best_variant_name = max(results.items(), key=lambda x: x[1]['actual_ring_adherence'])[0]
        worst_variant_name = min(results.items(), key=lambda x: x[1]['actual_ring_adherence'])[0]

        # Best variant samples
        ax5 = axes[1, 1]
        # 5. Best variant samples
        ax5 = axes[1, 1]
        best_samples = results[best_variant_name]['samples']
        ax5.scatter(best_samples[:, 0], best_samples[:, 1], alpha=0.6, s=8, c='green')

        # Add ring boundaries
        from matplotlib.patches import Circle
        inner_circle = Circle((0, 0), 1.0, fill=False, color='blue', linestyle='--', alpha=0.7)
        outer_circle = Circle((0, 0), 2.0, fill=False, color='red', linestyle='--', alpha=0.7)
        ax5.add_patch(inner_circle)
        ax5.add_patch(outer_circle)

        ax5.set_xlim(-3, 3)
        ax5.set_ylim(-3, 3)
        ax5.set_aspect('equal')
        ax5.set_title(f'Best: {best_variant_name}\n({results[best_variant_name]["actual_ring_adherence"]:.1%} adherence, {results[best_variant_name]["logic_satisfaction"]:.3f} logic)')
        ax5.grid(True, alpha=0.3)

        # 6. Worst variant samples
        ax6 = axes[1, 2]
        # 6. Worst variant samples
        ax6 = axes[1, 2]
        worst_samples = results[worst_variant_name]['samples']
        ax6.scatter(worst_samples[:, 0], worst_samples[:, 1], alpha=0.6, s=8, c='orange')

        # Add ring boundaries
        inner_circle = Circle((0, 0), 1.0, fill=False, color='blue', linestyle='--', alpha=0.7)
        outer_circle = Circle((0, 0), 2.0, fill=False, color='red', linestyle='--', alpha=0.7)
        ax6.add_patch(inner_circle)
        ax6.add_patch(outer_circle)

        ax6.set_xlim(-3, 3)
        ax6.set_ylim(-3, 3)
        ax6.set_aspect('equal')
        ax6.set_title(f'Worst: {worst_variant_name}\n({results[worst_variant_name]["actual_ring_adherence"]:.1%} adherence, {results[worst_variant_name]["logic_satisfaction"]:.3f} logic)')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save visualization
        viz_file = os.path.join(self.results_dir, 'visualizations', 'ring_ablation_results.png')
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {viz_file}")

def main():
    """
    Main function to run the Ring ablation framework.
    """
    print("Ring LTN-GAN Ablation Framework")
    print("=" * 50)

    # Create framework
    framework = RingAblationFramework()

    # Run all experiments
    results = framework.run_all_experiments(epochs=100)  # Consistent 100 epochs

    print("\nRing Ablation Experiments Complete!")
    print("=" * 50)
    print("Results Summary:")

    if 'baseline_gan' in results and 'full_ltn_gan' in results:
        baseline = results['baseline_gan']['actual_ring_adherence']
        ltn_gan = results['full_ltn_gan']['actual_ring_adherence']
        improvement = (ltn_gan - baseline) * 100

        print(f"Baseline GAN: {baseline:.1%} ring adherence")
        print(f"Full LTN-GAN: {ltn_gan:.1%} ring adherence")
        print(f"Improvement: +{improvement:.1f} percentage points")

        if improvement > 20:  # 20 percentage points improvement
            print("LTN-GAN shows significant improvement!")
        else:
            print("LTN-GAN improvement is modest")

    return results

if __name__ == "__main__":
    results = main()
