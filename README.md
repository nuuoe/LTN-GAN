# LTN-GAN: Logical Tensor Networks for Generative Adversarial Networks

This repository contains an implementation of LTN-GAN (Logical Tensor Networks GAN) across multiple datasets with ablation studies and analysis frameworks.

## File Structure

### Core LTN-GAN Implementations
- **`mnist_ltn_gan.py`** - MNIST digit generation with LTN constraints
- **`gaussian_ltn_gan.py`** - 2D Gaussian distribution generation with LTN constraints
- **`grid_ltn_gan.py`** - Grid pattern generation with LTN constraints
- **`ring_ltn_gan.py`** - Ring distribution generation with LTN constraints

### Ablation Study Frameworks
- **`mnist_ablation_framework.py`** - Systematic ablation study for MNIST
- **`gaussian_ablation_framework.py`** - Systematic ablation study for Gaussian
- **`grid_ablation_framework.py`** - Systematic ablation study for Grid
- **`ring_ablation_framework.py`** - Systematic ablation study for Ring

### Master Scripts
- **`run_all_core_experiments.py`** - Execute all core LTN-GAN experiments
- **`run_all_ablation_studies.py`** - Execute all ablation studies

## Quick Start

### Option 1: Run All Core Experiments
```bash
python run_all_core_experiments.py
```
This will execute all four core LTN-GAN implementations and generate comprehensive visualizations.

### Option 2: Run All Ablation Studies
```bash
python run_all_ablation_studies.py
```
This will execute all four ablation frameworks and generate detailed analysis reports.

### Option 3: Run Individual Experiments
```bash
# Core experiments
python mnist_ltn_gan.py
python gaussian_ltn_gan.py
python grid_ltn_gan.py
python ring_ltn_gan.py

# Ablation studies
python mnist_ablation_framework.py
python gaussian_ablation_framework.py
python grid_ablation_framework.py
python ring_ablation_framework.py
```

## Output Files

### Core Experiments Output
Each core experiment generates:
- **`{dataset}_comprehensive_comparison.png`** - 9-panel comprehensive analysis visualization
  - Panel 1: Generator Loss Comparison
  - Panel 2: Discriminator Loss Comparison
  - Panel 3: Baseline Generated Samples
  - Panel 4: LTN-GAN Generated Samples
  - Panel 5: Training Metrics Comparison
  - Panel 6: Performance Gap Analysis
  - Panel 7: Sample Quality Metrics
  - Panel 8: Constraint Satisfaction
  - Panel 9: Final Performance Summary

### Ablation Studies Output
Each ablation study generates:
- **`{dataset}_ablation_results/`** directory containing:
  - **`{dataset}_ablation_results.json`** - Raw experimental data
  - **`{dataset}_ablation_table.txt`** - Formatted results table
  - **`visualizations/{dataset}_ablation_results.png`** - Ablation visualization

## Ablation Study Variants

### MNIST Ablation Variants
- `baseline_gan` - Traditional GAN with templates but no LTN constraints
- `full_ltn_gan` - Complete LTN-GAN with templates and enhanced constraints
- `template_only` - Template guidance without LTN constraints
- `ltn_only` - LTN constraints without template guidance
- `weak_ltn` - LTN-GAN with moderate constraint pressure
- `strong_ltn` - LTN-GAN with maximum constraint pressure

### Gaussian Ablation Variants
- `baseline_gan` - Traditional GAN without any LTN constraints
- `full_ltn_gan` - Complete LTN-GAN using original trainer
- `no_constraints` - LTN architecture but constraints disabled
- `high_constraint` - LTN-GAN with high constraint pressure
- `fast_scheduling` - LTN-GAN with fast constraint scheduling
- `slow_scheduling` - LTN-GAN with slow constraint scheduling

### Grid Ablation Variants
- `baseline_gan` - Baseline GAN without LTN constraints
- `full_ltn_gan` - Full LTN-GAN with grid constraints
- `no_constraints` - LTN-GAN with disabled constraints
- `high_constraint` - LTN-GAN with high constraint weight
- `fast_scheduling` - LTN-GAN with fast scheduling
- `slow_scheduling` - LTN-GAN with slow scheduling

### Ring Ablation Variants
- `baseline_gan` - Baseline GAN without LTN constraints
- `full_ltn_gan` - Full LTN-GAN with ring constraints
- `no_constraints` - LTN-GAN with disabled constraints
- `no_hierarchical_weights` - LTN-GAN without hierarchical weighting
- `no_progressive_phases` - LTN-GAN without progressive phases
- `simple_constraints` - LTN-GAN with simple constraints

## Key Metrics

### MNIST Metrics
- **Quality Score** - Overall sample quality assessment
- **Coverage Score** - Digit class coverage
- **Digit Recognition** - Automated digit recognition accuracy
- **LTN Satisfaction** - Constraint satisfaction level
- **Template Dependency** - Template guidance effectiveness
- **Class Balance** - Distribution across digit classes

### Gaussian Metrics
- **Logic Satisfaction** - LTN constraint satisfaction
- **Gaussian Adherence** - Adherence to target Gaussian distribution
- **Statistical Quality** - Statistical similarity to target
- **Combined Quality** - Overall quality assessment
- **Intentionality Check** - Assessment of intentional vs accidental learning

### Grid Metrics
- **Grid Clustering Score** - Clustering quality assessment
- **Position Coverage** - Coverage of grid positions
- **Clustering Quality** - Overall clustering performance
- **Constraint Satisfaction** - LTN constraint satisfaction
- **Overall Performance** - Combined performance metric

### Ring Metrics
- **Ring Adherence** - Adherence to ring structure
- **Logic Satisfaction** - LTN constraint satisfaction
- **Balance Score** - Balance between inner/outer rings
- **Dead Zone Avoidance** - Avoidance of center dead zone

## Technical Details

### Dependencies
- PyTorch
- LTN (Logical Tensor Networks) - LTNtorch
- NumPy
- Matplotlib
- Scikit-learn (for some metrics)

### Environment Setup
```bash
# Create a conda environment with all dependencies and activate the conda environment
conda activate name-of-your-environment

# Or install dependencies manually
pip install torch torchvision torchaudio matplotlib numpy scikit-learn
pip install ltn
```

### Training Parameters
- **Epochs**: 100 for core experiments, 50-150 for ablation studies
- **Batch Size**: 64-128 depending on dataset
- **Learning Rate**: 0.0002 for generators, 0.0002 for discriminators
- **Constraint Weight**: 0.1-1.0 with scheduling



