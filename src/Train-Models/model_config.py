"""
Configuration module for neural network models.
Contains hyperparameters, architecture settings, and training configurations.
"""

from typing import Dict, Any

# Common configurations shared between models
COMMON_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 10,
    'validation_split': 0.1,
    'l2_reg': 0.001,
    'leaky_relu_alpha': 0.1,
    'optimizer': {
        'initial_learning_rate': 0.001,
        'decay_steps': 1000,
        'decay_rate': 0.9,
        'staircase': True
    }
}

# Money Line (ML) model configuration
ML_CONFIG = {
    'input_dim': None,  # Set dynamically based on input features
    'dense_layers': [
        {
            'units': 512,
            'dropout': 0.3,
            'batch_norm': True,
            'activation': 'leaky_relu'
        },
        {
            'units': 256,
            'dropout': 0.2,
            'batch_norm': True,
            'activation': 'leaky_relu'
        },
        {
            'units': 128,
            'dropout': 0.1,
            'batch_norm': True,
            'activation': 'leaky_relu'
        }
    ],
    'output_units': 2,
    'output_activation': 'softmax',
    'metrics': ['accuracy'],
    'loss': 'sparse_categorical_crossentropy'
}

# Under/Over (UO) model configuration
UO_CONFIG = {
    'input_dim': None,  # Set dynamically based on input features
    'dense_layers': [
        {
            'units': 256,
            'dropout': 0.3,
            'batch_norm': True,
            'activation': 'leaky_relu'
        },
        {
            'units': 128,
            'dropout': 0.2,
            'batch_norm': True,
            'activation': 'leaky_relu'
        },
        {
            'units': 64,
            'dropout': 0.1,
            'batch_norm': True,
            'activation': 'leaky_relu'
        }
    ],
    'output_units': 3,  # Under, Over, Push
    'output_activation': 'softmax',
    'metrics': ['accuracy'],
    'loss': 'sparse_categorical_crossentropy'
}

# Data split configuration
DATA_SPLIT_CONFIG = {
    'train_end_date': '2021-01-01',
    'val_end_date': '2022-01-01',
    # Everything after val_end_date is test data
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'rolling_windows': [3, 5, 10],
    'relative_metrics': True,
    'streak_features': True
}

def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get the complete configuration for a specific model type.
    
    Args:
        model_type: Either 'ml' for Money Line or 'uo' for Under/Over
        
    Returns:
        Dictionary containing the complete model configuration
    """
    if model_type.lower() == 'ml':
        specific_config = ML_CONFIG
    elif model_type.lower() == 'uo':
        specific_config = UO_CONFIG
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Merge with common config
    config = {
        **COMMON_CONFIG,
        **specific_config
    }
    
    return config

def update_input_dim(config: Dict[str, Any], input_dim: int) -> Dict[str, Any]:
    """
    Update the input dimension in the configuration.
    
    Args:
        config: Model configuration dictionary
        input_dim: Input feature dimension
        
    Returns:
        Updated configuration dictionary
    """
    config['input_dim'] = input_dim
    return config 