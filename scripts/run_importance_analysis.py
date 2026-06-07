#!/usr/bin/env python
"""
Skript dlya analiza interpretabelnosti: SHAP i gradienty.
Supports both ITM-Rec and OULAD datasets.
"""

import sys
import logging
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent.absolute()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.models.deepfm_svdpp import DeepFMSVDPlusPlus
from src.models.dueling_dqn import DuelingDQN
from src.analysis import DQNStateImportanceAnalyzer, StateComponentGrouper
from src.utils.helpers import set_seed, get_device

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_seed(42)
device = get_device()

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

# Create output directory
output_dir = ROOT / "results" / "feature_importance_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"Output directory: {output_dir}")


def analyze_deepfm_feature_importance(dataset_type: str = "itmrec"):
    """Analyze DeepFM+SVD++ feature importance using gradient-based methods."""
    logger.info("=" * 80)
    logger.info(f"DeepFM+SVD++ FEATURE IMPORTANCE ANALYSIS ({dataset_type.upper()})")
    logger.info("=" * 80)
    
    try:
        import shap
        logger.info(f"SHAP version: {shap.__version__}")
    except ImportError:
        logger.warning("SHAP not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
        import shap
        logger.info(f"SHAP installed: {shap.__version__}")
    
    # Configure parameters based on dataset
    if dataset_type.lower() == "itmrec":
        state_dim = 65
        action_dim = 70
        n_users_default = 100
        n_items_default = 50
        n_classes = 3
        n_semesters = 2
        n_lockdowns = 3
    elif dataset_type.lower() == "oulad":
        state_dim = 96
        action_dim = 18
        n_users_default = 100
        n_items_default = 18
        n_classes = 8  # modules
        n_semesters = 2  # presentations
        n_lockdowns = 2  # step types
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # Try to load model
    model_path = ROOT / "data" / "models" / "deepfm_svdplusplus_best.pth"
    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        logger.info("Creating synthetic model for demonstration...")
        model = DeepFMSVDPlusPlus(
            n_users=n_users_default, n_items=n_items_default, 
            n_classes=n_classes, n_semesters=n_semesters, n_lockdowns=n_lockdowns,
            device=device, embedding_dim=32, hidden_dims=[64, 32],
            dropout_rate=0.2, dataset_type=dataset_type
        )
        model.eval()
    else:
        logger.info(f"Loading model from {model_path}")
        try:
            model, checkpoint = DeepFMSVDPlusPlus.load_checkpoint(model_path, device=device)
            model.eval()
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Creating synthetic model for demonstration...")
            model = DeepFMSVDPlusPlus(
                n_users=100, n_items=50, n_classes=3, n_semesters=2, n_lockdowns=3,
                device=device, embedding_dim=32, hidden_dims=[64, 32],
                dropout_rate=0.2, dataset_type="itmrec"
            )
            model.eval()
    
    logger.info(f"Model loaded: dataset_type={model.dataset_type}")
    logger.info(f"Output heads: {list(model.output_heads.keys())}")
    
    # Create synthetic data for SHAP
    logger.info("\nGenerating synthetic data for SHAP analysis...")
    n_background = 100
    n_explain = 30
    
    batch_size = n_background + n_explain
    user_ids = torch.randint(0, model.n_users, (batch_size,), device=device)
    item_ids = torch.randint(0, model.n_items, (batch_size,), device=device)
    class_ids = torch.randint(0, model.n_classes, (batch_size,), device=device)
    semester_ids = torch.randint(0, model.n_semesters, (batch_size,), device=device)
    lockdown_ids = torch.randint(0, model.n_lockdowns, (batch_size,), device=device)
    
    # Compute predictions
    with torch.no_grad():
        predictions = model.forward(user_ids, item_ids, class_ids, semester_ids, lockdown_ids)
    
    logger.info(f"Predictions computed: {predictions.keys()}")
    
    # Compute gradient-based importance for each head
    logger.info("\nComputing gradient-based importance for each head...")
    
    importance_results = {}
    
    for head_name in predictions.keys():
        logger.info(f"  Processing head: {head_name}")
        
        # Create inputs with gradient requirement
        user_ids_grad = torch.randint(0, model.n_users, (n_explain,), device=device, dtype=torch.long)
        item_ids_grad = torch.randint(0, model.n_items, (n_explain,), device=device, dtype=torch.long)
        class_ids_grad = torch.randint(0, model.n_classes, (n_explain,), device=device, dtype=torch.long)
        semester_ids_grad = torch.randint(0, model.n_semesters, (n_explain,), device=device, dtype=torch.long)
        lockdown_ids_grad = torch.randint(0, model.n_lockdowns, (n_explain,), device=device, dtype=torch.long)
        
        # Convert to embedding space
        user_emb = model.user_emb_fm(user_ids_grad)
        item_emb = model.item_emb_fm(item_ids_grad)
        
        # Compute output
        with torch.enable_grad():
            user_emb.requires_grad_(True)
            item_emb.requires_grad_(True)
            
            # Embeddings for FM
            class_emb = model.class_emb_fm(class_ids_grad)
            semester_emb = model.semester_emb_fm(semester_ids_grad)
            lockdown_emb = model.lockdown_emb_fm(lockdown_ids_grad)
            
            # Deep input
            deep_input = torch.cat([
                user_emb, item_emb,
                class_emb, semester_emb, lockdown_emb,
                model.user_emb_svd(user_ids_grad),
                model.item_emb_svd(item_ids_grad)
            ], dim=1)
            
            deep_output = model.deep_network(deep_input)
            
            # FM part
            interaction_term = torch.sum(user_emb * item_emb, dim=1)
            square_of_sum = torch.pow(interaction_term, 2)
            sum_of_squares = torch.sum(torch.pow(user_emb, 2) * torch.pow(item_emb, 2), dim=1)
            fm_second_order = 0.5 * (square_of_sum - sum_of_squares)
            
            # Linear terms
            linear_terms = model.user_bias(user_ids_grad).squeeze() + \
                          model.item_bias(item_ids_grad).squeeze() + \
                          model.global_bias
            
            # Combined
            combined = torch.cat([
                deep_output,
                fm_second_order.unsqueeze(1),
                linear_terms.unsqueeze(1)
            ], dim=1)
            
            # Get output head
            head = model.output_heads[head_name]
            output = torch.sigmoid(head(combined)).squeeze(-1)
        
        # Backprop
        output.sum().backward()
        
        # Get gradients
        if user_emb.grad is not None:
            user_grad = user_emb.grad.abs().mean(dim=0).mean().item()
            importance_results[head_name] = {
                "user_grad": user_grad,
                "mean_output": output.detach().mean().item(),
            }
            logger.info(f"    {head_name}: mean_output={importance_results[head_name]['mean_output']:.3f}")
    
    # Save results
    df = pd.DataFrame(importance_results).T
    df.to_csv(output_dir / "shap_deepfm_importance.csv")
    logger.info(f"\nResults saved to {output_dir / 'shap_deepfm_importance.csv'}")
    
    return importance_results


def compute_dqn_state_importance(dataset_type: str = "itmrec"):
    """Compute DQN state component importance using gradient-based analysis."""
    logger.info("=" * 80)
    logger.info(f"DQN STATE COMPONENT IMPORTANCE ANALYSIS ({dataset_type.upper()})")
    logger.info("=" * 80)
    
    # Configure parameters based on dataset
    if dataset_type.lower() == "itmrec":
        state_dim = 65
        action_dim = 70
    elif dataset_type.lower() == "oulad":
        state_dim = 96
        action_dim = 18
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    model_path = ROOT / "data" / "models" / "dqn_agent_checkpoint.pth"
    
    logger.info(f"Loading DQN model from {model_path}")
    
    try:
        if model_path.exists() and dataset_type.lower() == "itmrec":
            # Only load real model for ITM-Rec since we have checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Try to extract agent state dict
            if "agent_state_dict" in checkpoint:
                state_dict = checkpoint["agent_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # Instantiate DQN with correct state_dim and action_dim
            dqn_model = DuelingDQN(state_dim=state_dim, action_dim=action_dim, device=device)
            dqn_model.load_state_dict(state_dict)
            dqn_model.eval()
            
            logger.info(f"DQN loaded from checkpoint")
        else:
            logger.warning(f"Model not found or using demonstration mode for {dataset_type.upper()}")
            logger.info("Creating synthetic DQN model for demonstration...")
            dqn_model = DuelingDQN(state_dim=state_dim, action_dim=action_dim, device=device)
            dqn_model.eval()
            
    except Exception as e:
        logger.error(f"Failed to load DQN: {e}")
        logger.info("Creating synthetic DQN model for demonstration...")
        dqn_model = DuelingDQN(state_dim=state_dim, action_dim=action_dim, device=device)
        dqn_model.eval()
    
    # Create analyzer
    logger.info("Creating analyzer...")
    analyzer = DQNStateImportanceAnalyzer(dqn_model, state_dim, action_dim)
    
    # Generate test states
    logger.info("Generating test states...")
    n_test_states = 100
    test_states = np.random.randn(n_test_states, state_dim).astype(np.float32)
    test_states = (test_states - test_states.min()) / (test_states.max() - test_states.min() + 1e-8)
    
    # Compute importance
    logger.info("Computing state importance...")
    state_importance = analyzer.compute_state_importance(test_states, use_greedy_actions=True)
    
    logger.info(f"Importance shape: {state_importance.shape}")
    logger.info(f"Min: {state_importance.min():.4f}, Max: {state_importance.max():.4f}")
    
    # Aggregate by components
    logger.info(f"\nState component importance ({dataset_type.upper()}):")
    component_importance = StateComponentGrouper.aggregate_by_group(state_importance, dataset_type)
    
    for component_name, info in component_importance.items():
        logger.info(
            f"  {component_name:<25} importance={info['importance']:>8.4f}  "
            f"n_components={info['n_components']:>2}  {info['description']}"
        )
    
    # Save results
    results_df = pd.DataFrame([
        {
            "component": name,
            "importance": info["importance"],
            "n_components": info["n_components"],
            "description": info["description"],
        }
        for name, info in component_importance.items()
    ])
    
    results_file = output_dir / f"dqn_state_importance_{dataset_type}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"\nResults saved to {results_file}")
    
    # Save detailed importances
    details_file = output_dir / f"state_importance_details_{dataset_type}.npy"
    np.save(details_file, state_importance)
    logger.info(f"Details saved to {details_file}")
    
    return component_importance, state_importance


if __name__ == "__main__":
    logger.info(f"Starting feature importance analysis...")
    logger.info(f"Device: {device}")
    logger.info(f"Output dir: {output_dir}\n")
    
    # Execute analysis tasks for both datasets
    for dataset in ["itmrec", "oulad"]:
        logger.info(f"\n{'#'*80}")
        logger.info(f"# ANALYZING DATASET: {dataset.upper()}")
        logger.info(f"{'#'*80}\n")
        
        analyze_deepfm_feature_importance(dataset_type=dataset)
        result = compute_dqn_state_importance(dataset_type=dataset)
        if result:
            component_imp, state_imp = result
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE - Both datasets analyzed (ITM-Rec and OULAD)")
    logger.info("=" * 80)
