import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Input
from tensorflow.keras.regularizers import l2
from src.Utils.evaluation import evaluate_multiclass_classification, print_evaluation_report

def time_based_split(df, train_end_date, val_end_date):
    """Split data based on dates to prevent future data leakage."""
    train_data = df[pd.to_datetime(df['Date']) <= train_end_date].copy()
    val_data = df[(pd.to_datetime(df['Date']) > train_end_date) & 
                  (pd.to_datetime(df['Date']) <= val_end_date)].copy()
    test_data = df[pd.to_datetime(df['Date']) > val_end_date].copy()
    
    print(f"Train set size: {len(train_data)} ({train_data['Date'].min()} to {train_data['Date'].max()})")
    print(f"Validation set size: {len(val_data)} ({val_data['Date'].min()} to {val_data['Date'].max()})")
    print(f"Test set size: {len(test_data)} ({test_data['Date'].min()} to {test_data['Date'].max()})")
    
    return train_data, val_data, test_data

# Setup logging and callbacks
current_time = str(time.time())
tensorboard = TensorBoard(log_dir='../../Logs/{}'.format(current_time))
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('../../Models/Trained-Model-OU-' + current_time, save_best_only=True, monitor='val_loss', mode='min')

# Load data
dataset = "dataset_2012-24_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

# Split data based on time
train_end_date = pd.to_datetime('2021-01-01')
val_end_date = pd.to_datetime('2022-01-01')
train_data, val_data, test_data = time_based_split(data, train_end_date, val_end_date)

def prepare_data(df):
    """Prepare data for training/validation/testing."""
    ou = df['OU-Cover']
    total = df['OU']
    df = df.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'], axis=1)
    df['OU'] = np.asarray(total)
    data_values = df.values.astype(float)
    x = tf.keras.utils.normalize(data_values, axis=1)
    y = np.asarray(ou)
    return x, y

# Prepare datasets
x_train, y_train = prepare_data(train_data)
x_val, y_val = prepare_data(val_data)
x_test, y_test = prepare_data(test_data)

# Define learning rate schedule
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Define the model with modern layers and regularization
input_shape = x_train.shape[1:]
model = tf.keras.models.Sequential([
    Input(shape=input_shape),
    BatchNormalization(),
    Dense(256, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    
    Dense(128, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    
    Dense(64, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.1),
    
    Dense(3, activation='softmax')  # 3 classes for Under/Over/Push
])

# Calculate class weights to handle imbalance
class_weights = dict(zip(
    np.unique(y_train),
    1 / np.bincount(y_train.astype(int))
))

model.compile(optimizer=optimizer, 
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Train with separate validation set
history = model.fit(
    x_train, y_train,
    epochs=50,
    validation_data=(x_val, y_val),
    batch_size=32,
    class_weight=class_weights,
    callbacks=[tensorboard, earlyStopping, mcp_save]
)

# Get predictions for evaluation
y_pred_proba = model.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Evaluate using our comprehensive metrics
metrics = evaluate_multiclass_classification(y_test, y_pred, y_pred_proba)
print_evaluation_report(metrics, model_type="multiclass")

# Print training history summary
print("\nTraining History:")
print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
print('Done')
