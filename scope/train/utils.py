from typing import *
import fnmatch

import sympy
import torch
import torch.nn as nn


def any_match(s: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(s, pat) for pat in patterns)


def build_optimizer(model: nn.Module, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
    named_param_groups = [
        {
            k: p for k, p in model.named_parameters() if any_match(k, param_group_config['params']['include']) and not any_match(k, param_group_config['params'].get('exclude', []))
        } for param_group_config in optimizer_config['params']
    ]
    excluded_params = [k for k, p in model.named_parameters() if p.requires_grad and not any(k in named_params for named_params in named_param_groups)]
    assert len(excluded_params) == 0, f'The following parameters require grad but are excluded from the optimizer: {excluded_params}'
    optimizer_cls = getattr(torch.optim, optimizer_config['type'])
    optimizer = optimizer_cls([
        {
            **param_group_config,
            'params': list(params.values()), 
        } for param_group_config, params in zip(optimizer_config['params'], named_param_groups)
    ])
    return optimizer


def parse_lr_lambda(s: str) -> Callable[[int], float]:
    epoch = sympy.symbols('epoch')
    lr_lambda = sympy.sympify(s)
    return sympy.lambdify(epoch, lr_lambda, 'math')


def build_lr_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    if scheduler_config['type'] == "SequentialLR":
        child_schedulers = [
            build_lr_scheduler(optimizer, child_scheduler_config)
                for child_scheduler_config in scheduler_config['params']['schedulers']
        ]
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=child_schedulers, milestones=scheduler_config['params']['milestones'])
    elif scheduler_config['type'] == "LambdaLR":
        lr_lambda = scheduler_config['params']['lr_lambda']
        if isinstance(lr_lambda, str):
            lr_lambda = parse_lr_lambda(lr_lambda)
        elif isinstance(lr_lambda, list):
            lr_lambda = [parse_lr_lambda(l) for l in lr_lambda]
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )
    else:
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config['type'])
        scheduler = scheduler_cls(optimizer, **scheduler_config.get('params', {}))
    return scheduler


def adjust_config(config, datasets, batch_size_forward, gradient_accumulation_steps, num_processes, images_per_sample, epochs):
    """
    Adjusts training parameters to maintain consistency when transitioning from single-frame to multi-frame model.
    
    Args:
        config: The original configuration dictionary
        datasets: List of datasets to calculate total samples
        batch_size_forward: Batch size for each forward pass
        gradient_accumulation_steps: Number of steps to accumulate gradients
        num_processes: Number of processes for distributed training
        images_per_sample: Number of frames per sample in multi-frame model
        epochs: Number of epochs to train
        
    Returns:
        Adjusted configuration dictionary
    """
    import copy
    
    # Calculate total dataset size
    total_samples = sum(len(dataset) for dataset in datasets)
    
    # Calculate iterations per epoch (before distribution)
    iters_per_epoch = total_samples / batch_size_forward
    
    # Calculate total iterations (accounting for distribution)
    new_total_iterations = epochs * iters_per_epoch / num_processes
    
    # Original parameters from first version of the model
    original_batch_size_forward = 2
    original_gradient_accumulation_steps = 2
    original_num_processes = 8
    original_images_per_sample = 1  # Single-frame model
    original_total_iterations = 1000000
    
    # Calculate batch size ratios
    original_batch_size = original_batch_size_forward * original_gradient_accumulation_steps * original_num_processes * original_images_per_sample
    new_batch_size = batch_size_forward * gradient_accumulation_steps * num_processes * images_per_sample
    batch_size_ratio = new_batch_size / original_batch_size
    
    # Calculate iteration ratio
    iteration_ratio = original_total_iterations / new_total_iterations if new_total_iterations > 0 else 1.0
    
    print(f"Adjusting config parameters:")
    print(f"  Original batch size: {original_batch_size}")
    print(f"  New batch size: {new_batch_size}")
    print(f"  Batch size ratio: {batch_size_ratio:.4f}")
    print(f"  Original iterations: {original_total_iterations}")
    print(f"  Estimated new iterations: {new_total_iterations:.0f}")
    print(f"  Iteration ratio: {iteration_ratio:.4f}")
    
    # Create a deep copy to avoid modifying the original
    adjusted_config = copy.deepcopy(config)
    
    # 1. Adjust learning rates based on batch size ratio (linear scaling rule)
    print("\nAdjusting learning rates:")
    for i, param_group in enumerate(adjusted_config['optimizer']['params']):
        original_lr = param_group['lr']
        adjusted_lr = original_lr * (batch_size_ratio ** 0.5)
        param_group['lr'] = adjusted_lr
        print(f"  Group {i}: {original_lr:.1e} → {adjusted_lr:.1e}")
    
    # 2. Adjust scheduler parameters based on iteration ratio
    if 'low_resolution_training_steps' in adjusted_config:
        original_steps = adjusted_config['low_resolution_training_steps']
        adjusted_steps = max(1, int(original_steps / iteration_ratio))
        adjusted_config['low_resolution_training_steps'] = adjusted_steps
        print(f"\nAdjusting low_resolution_training_steps: {original_steps} → {adjusted_steps}")
    
    # 3. Adjust scheduler milestones and step sizes
    if 'lr_scheduler' in adjusted_config and 'params' in adjusted_config['lr_scheduler']:
        print("\nAdjusting learning rate scheduler parameters:")
        
        if 'milestones' in adjusted_config['lr_scheduler']['params']:
            original_milestones = adjusted_config['lr_scheduler']['params']['milestones']
            adjusted_milestones = [max(1, int(milestone / iteration_ratio)) for milestone in original_milestones]
            adjusted_config['lr_scheduler']['params']['milestones'] = adjusted_milestones
            print(f"  Scheduler milestones: {original_milestones} → {adjusted_milestones}")
        
        if 'schedulers' in adjusted_config['lr_scheduler']['params']:
            for i, scheduler in enumerate(adjusted_config['lr_scheduler']['params']['schedulers']):
                if scheduler['type'] == 'StepLR' and 'step_size' in scheduler['params']:
                    original_step_size = scheduler['params']['step_size']
                    adjusted_step_size = max(1, int(original_step_size / iteration_ratio))
                    scheduler['params']['step_size'] = adjusted_step_size
                    print(f"  StepLR step_size: {original_step_size} → {adjusted_step_size}")
                
                # Adjust LambdaLR expressions that contain specific values
                if scheduler['type'] == 'LambdaLR' and 'lr_lambda' in scheduler['params']:
                    if isinstance(scheduler['params']['lr_lambda'], list):
                        for j, lambda_expr in enumerate(scheduler['params']['lr_lambda']):
                            if isinstance(lambda_expr, str) and 'epoch - 1000' in lambda_expr:
                                # new_decay_point = max(1, int(1000 / iteration_ratio))
                                # new_lambda_expr = lambda_expr.replace('epoch - 1000', f'epoch - {new_decay_point}')
                                new_decay_point = max(1, int(1000 / iteration_ratio))
                                new_lambda_expr = lambda_expr.replace('epoch - 1000', f'epoch - {new_decay_point}')
                                new_lambda_expr = new_lambda_expr.replace('/ 1000', f'/{new_decay_point}')
                                scheduler['params']['lr_lambda'][j] = new_lambda_expr
                                print(f"  LambdaLR expression adjusted: '{lambda_expr}' → '{new_lambda_expr}'")
    
    return adjusted_config