"""
Utility module for building neural network models based on configurations.
"""

from typing import Dict, Any

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

def build_dense_layer(config: Dict[str, Any], input_shape: tuple = None) -> Sequential:
    """
    Build a dense layer based on configuration.
    
    Args:
        config: Layer configuration dictionary
        input_shape: Optional input shape for first layer
        
    Returns:
        List of Keras layers
    """
    layers = []
    
    # Add input layer if shape is provided
    if input_shape is not None:
        layers.append(Input(shape=input_shape))
    
    # Add batch normalization before the dense layer
    if config.get('batch_norm', True):
        layers.append(BatchNormalization())
    
    # Add dense layer with L2 regularization
    dense = Dense(
        units=config['units'],
        kernel_regularizer=l2(0.001)
    )
    layers.append(dense)
    
    # Add batch normalization after the dense layer
    if config.get('batch_norm', True):
        layers.append(BatchNormalization())
    
    # Add activation
    if config.get('activation') == 'leaky_relu':
        layers.append(LeakyReLU(alpha=0.1))
    
    # Add dropout if specified
    if config.get('dropout', 0) > 0:
        layers.append(Dropout(config['dropout']))
    
    return layers

def build_model(config: Dict[str, Any]) -> Sequential:
    """
    Build a complete model based on configuration.
    
    Args:
        config: Complete model configuration dictionary
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential()
    
    # Add input layer
    input_shape = (config['input_dim'],)
    
    # Build hidden layers
    for i, layer_config in enumerate(config['dense_layers']):
        layers = build_dense_layer(
            layer_config,
            input_shape=input_shape if i == 0 else None
        )
        for layer in layers:
            model.add(layer)
    
    # Add output layer
    model.add(Dense(
        units=config['output_units'],
        activation=config['output_activation']
    ))
    
    # Configure optimizer with learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=config['optimizer']['initial_learning_rate'],
        decay_steps=config['optimizer']['decay_steps'],
        decay_rate=config['optimizer']['decay_rate'],
        staircase=config['optimizer']['staircase']
    )
    optimizer = Adam(learning_rate=lr_schedule)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=config['loss'],
        metrics=config['metrics']
    )
    
    return model

def create_callbacks(config: Dict[str, Any], model_type: str) -> list:
    """
    Create training callbacks based on configuration.
    
    Args:
        config: Model configuration dictionary
        model_type: Type of model ('ml' or 'uo')
        
    Returns:
        List of Keras callbacks
    """
    current_time = str(int(tf.timestamp()))
    
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=f'../../Logs/{current_time}'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping_patience'],
            mode='min',
            verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'../../Models/Trained-Model-{model_type.upper()}-{current_time}',
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
    ]
    
    return callbacks 