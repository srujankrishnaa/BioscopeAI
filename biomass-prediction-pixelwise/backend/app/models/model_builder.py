# CNN+LSTM Model Building for Above Ground Biomass Prediction
# Enhanced version with better architecture and training process

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, TimeDistributed, 
                                    Flatten, LSTM, Dense, Dropout, BatchNormalization, 
                                    concatenate, Reshape, Conv2DTranspose, UpSampling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                        ModelCheckpoint, LearningRateScheduler)
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from tensorflow.keras import backend as K
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom metrics
def r2_metric(y_true, y_pred):
    """Custom R² metric for Keras"""
    SS_res = K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - SS_res / (SS_tot + K.epsilon())

def rmse_metric(y_true, y_pred):
    """Custom RMSE metric for Keras"""
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# 2.1 Enhanced Model Architecture
def build_cnn_lstm_model(input_shape, num_filters=(32, 64, 128), lstm_units=128, dropout_rate=0.3):
    """
    Enhanced CNN+LSTM model with spatial preservation
    Uses a U-Net like decoder to preserve spatial details
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Encoder (CNN) - spatial feature extraction
    # Level 1
    conv1 = TimeDistributed(Conv2D(num_filters[0], (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)))(inputs)
    conv1 = TimeDistributed(BatchNormalization())(conv1)
    conv1 = TimeDistributed(Conv2D(num_filters[0], (3, 3), activation='relu', padding='same'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D((2, 2)))(conv1)
    
    # Level 2
    conv2 = TimeDistributed(Conv2D(num_filters[1], (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)))(pool1)
    conv2 = TimeDistributed(BatchNormalization())(conv2)
    conv2 = TimeDistributed(Conv2D(num_filters[1], (3, 3), activation='relu', padding='same'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D((2, 2)))(conv2)
    
    # Level 3
    conv3 = TimeDistributed(Conv2D(num_filters[2], (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)))(pool2)
    conv3 = TimeDistributed(BatchNormalization())(conv3)
    conv3 = TimeDistributed(Conv2D(num_filters[2], (3, 3), activation='relu', padding='same'))(conv3)
    pool3 = TimeDistributed(MaxPooling2D((2, 2)))(conv3)
    
    # Flatten spatial dimensions for LSTM
    flat = TimeDistributed(Flatten())(pool3)
    
    # LSTM layers for temporal modeling
    lstm1 = LSTM(lstm_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)(flat)
    lstm2 = LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate)(lstm1)
    
    # Dense layers
    dense1 = Dense(256, activation='relu')(lstm2)
    dense1 = Dropout(dropout_rate)(dense1)
    dense2 = Dense(128, activation='relu')(dense1)
    dense2 = Dropout(dropout_rate)(dense2)
    
    # Calculate the size after flattening
    # After 3 pooling layers, spatial dimensions are reduced by 8x
    reduced_height = input_shape[1] // 8
    reduced_width = input_shape[2] // 8
    reduced_features = num_filters[2]  # Number of filters at the deepest level
    
    # Reshape to spatial dimensions for decoder
    reshape = Reshape((reduced_height, reduced_width, reduced_features))(dense2)
    
    # Decoder - upsampling to restore spatial resolution
    up3 = TimeDistributed(Conv2DTranspose(num_filters[2], (3, 3), strides=(2, 2), padding='same'))(reshape)
    up3 = TimeDistributed(BatchNormalization())(up3)
    up3 = TimeDistributed(Conv2D(num_filters[2], (3, 3), activation='relu', padding='same'))(up3)
    
    # Skip connection from conv3
    merge3 = concatenate([up3, conv3])
    
    up2 = TimeDistributed(Conv2DTranspose(num_filters[1], (3, 3), strides=(2, 2), padding='same'))(merge3)
    up2 = TimeDistributed(BatchNormalization())(up2)
    up2 = TimeDistributed(Conv2D(num_filters[1], (3, 3), activation='relu', padding='same'))(up2)
    
    # Skip connection from conv2
    merge2 = concatenate([up2, conv2])
    
    up1 = TimeDistributed(Conv2DTranspose(num_filters[0], (3, 3), strides=(2, 2), padding='same'))(merge2)
    up1 = TimeDistributed(BatchNormalization())(up1)
    up1 = TimeDistributed(Conv2D(num_filters[0], (3, 3), activation='relu', padding='same'))(up1)
    
    # Final convolution to get single channel output
    outputs = TimeDistributed(Conv2D(1, (1, 1), activation='linear'))(up1)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with custom metrics
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', rmse_metric, r2_metric]
    )
    
    return model

# 2.2 Enhanced Model Training
def train_model(sequences, targets, model_save_path, val_split=0.2, batch_size=32, epochs=50):
    """
    Enhanced training function with data augmentation and better evaluation
    """
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, targets, test_size=val_split, random_state=42
    )
    
    # Get input shape
    input_shape = X_train.shape[1:]
    
    # Build model
    model = build_cnn_lstm_model(input_shape)
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7, monitor='val_loss'),
        ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss')
    ]
    
    # Learning rate scheduler
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        elif epoch < 20:
            return lr * 0.5
        else:
            return lr * 0.1
    
    callbacks.append(LearningRateScheduler(lr_scheduler))
    
    # Data augmentation function for satellite imagery
    def augment_data(sequences, targets):
        """Apply data augmentation to sequences and targets"""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            sequences = np.flip(sequences, axis=3)  # Flip width dimension
            targets = np.flip(targets, axis=2)    # Flip width dimension
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            sequences = np.flip(sequences, axis=2)  # Flip height dimension
            targets = np.flip(targets, axis=1)    # Flip height dimension
        
        # Random rotation (90, 180, or 270 degrees)
        if np.random.rand() > 0.7:
            k = np.random.randint(1, 4)  # 1, 2, or 3 rotations
            sequences = np.rot90(sequences, k=k, axes=(2, 3))
            targets = np.rot90(targets, k=k, axes=(1, 2))
        
        return sequences, targets
    
    # Custom data generator with augmentation
    def data_generator(X, y, batch_size, augment=False):
        """Generate batches of data with optional augmentation"""
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            if augment:
                X_batch, y_batch = augment_data(X_batch, y_batch)
            
            yield X_batch, y_batch
    
    # Create generators
    train_generator = data_generator(X_train, y_train, batch_size, augment=True)
    val_generator = data_generator(X_val, y_val, batch_size, augment=False)
    
    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# 2.3 Enhanced Model Evaluation
def evaluate_model(model, X_test, y_test, output_dir='./plots'):
    """
    Enhanced model evaluation with additional metrics and visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Flatten arrays for metric calculation
    y_true_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove NaN values
    mask = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
    y_true_flat = y_true_flat[mask]
    y_pred_flat = y_pred_flat[mask]
    
    # Calculate metrics
    r2 = r2_score(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    
    logger.info(f"R² Score: {r2:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    
    # Plot scatter plot of predictions vs actuals
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.5)
    plt.plot([y_true_flat.min(), y_true_flat.max()], [y_true_flat.min(), y_true_flat.max()], 'r--')
    plt.xlabel('Actual Biomass')
    plt.ylabel('Predicted Biomass')
    plt.title(f'Predictions vs Actuals (R² = {r2:.4f})')
    plt.savefig(os.path.join(output_dir, 'predictions_vs_actuals.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot spatial comparison for a sample
    sample_idx = 0
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(y_test[sample_idx, :, :, 0], cmap='YlGn')
    plt.colorbar(label='Actual Biomass')
    plt.title('Actual Biomass')
    
    plt.subplot(1, 2, 2)
    plt.imshow(y_pred[sample_idx, :, :, 0], cmap='YlGn')
    plt.colorbar(label='Predicted Biomass')
    plt.title('Predicted Biomass')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error distribution
    errors = y_true_flat - y_pred_flat
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot biomass value distribution comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(y_true_flat, bins=50, kde=True, color='blue', label='Actual')
    plt.title('Distribution of Actual Biomass')
    plt.xlabel('Biomass Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(y_pred_flat, bins=50, kde=True, color='orange', label='Predicted')
    plt.title('Distribution of Predicted Biomass')
    plt.xlabel('Biomass Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'biomass_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot error heatmap
    # Calculate spatial errors for the sample
    spatial_errors = y_test[sample_idx, :, :, 0] - y_pred[sample_idx, :, :, 0]
    plt.figure(figsize=(8, 6))
    plt.imshow(spatial_errors, cmap='RdBu_r', vmin=-np.max(np.abs(spatial_errors)), vmax=np.max(np.abs(spatial_errors)))
    plt.colorbar(label='Prediction Error')
    plt.title('Spatial Error Distribution')
    plt.savefig(os.path.join(output_dir, 'spatial_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }

# 2.4 Hyperparameter Tuning
def tune_hyperparameters(sequences, targets, max_trials=5, executions_per_trial=2):
    """
    Perform hyperparameter tuning using Keras Tuner
    """
    try:
        import kerastuner as kt
    except ImportError:
        logger.error("Keras Tuner not installed. Skipping hyperparameter tuning.")
        return None, None
    
    def build_model(hp):
        # Define hyperparameter search space
        num_filters_1 = hp.Int('num_filters_1', 16, 64, step=16)
        num_filters_2 = hp.Int('num_filters_2', 32, 128, step=32)
        num_filters_3 = hp.Int('num_filters_3', 64, 256, step=64)
        lstm_units = hp.Int('lstm_units', 64, 256, step=64)
        dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        
        # Build model with tunable parameters
        input_shape = sequences.shape[1:]
        model = build_cnn_lstm_model(
            input_shape,
            num_filters=(num_filters_1, num_filters_2, num_filters_3),
            lstm_units=lstm_units,
            dropout_rate=dropout_rate
        )
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', rmse_metric, r2_metric]
        )
        
        return model
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, targets, test_size=0.2, random_state=42
    )
    
    # Create tuner
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='tuner_results',
        project_name='biomass_prediction'
    )
    
    # Perform tuning
    tuner.search(
        X_train, y_train,
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=5)]
    )
    
    # Get best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return best_model, best_hyperparameters

# 2.5 Model Quantization
def quantize_model(model_path, quantized_model_path):
    """
    Quantize model for efficient deployment
    """
    logger.info(f"Quantizing model from {model_path} to {quantized_model_path}")
    
    # Load trained model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TensorFlow Lite model with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Quantize model
    quantized_model = converter.convert()
    
    # Save quantized model
    with open(quantized_model_path, 'wb') as f:
        f.write(quantized_model)
    
    logger.info(f"Model quantized and saved to {quantized_model_path}")
    return quantized_model_path

# 2.6 Enhanced Model Training Execution
def main():
    """Main training execution function with enhanced features"""
    # Create necessary directories
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./plots', exist_ok=True)
    
    # Load preprocessed data
    try:
        sequences = np.load('./data/processed/sequences.npy')
        targets = np.load('./data/processed/targets.npy')
        logger.info(f"Loaded data - Sequences shape: {sequences.shape}, Targets shape: {targets.shape}")
    except FileNotFoundError:
        logger.error("Error: Preprocessed data not found. Please run data_preprocessing.py first.")
        return
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, targets, test_size=0.2, random_state=42
    )
    
    # Option: Perform hyperparameter tuning (uncomment to use)
    # logger.info("Performing hyperparameter tuning...")
    # best_model, best_hyperparameters = tune_hyperparameters(X_train, y_train, max_trials=5)
    # if best_model:
    #     best_model.save('./models/biomass_cnn_lstm_tuned.h5')
    #     logger.info("Best hyperparameters: {}".format(best_hyperparameters.values))
    
    # Train model
    logger.info("Training model...")
    model, history = train_model(
        X_train, y_train,
        model_save_path='./models/biomass_cnn_lstm.h5',
        val_split=0.2,
        batch_size=16,
        epochs=50
    )
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # MAE
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # RMSE
    plt.subplot(2, 2, 3)
    plt.plot(history.history['rmse_metric'], label='Training RMSE')
    plt.plot(history.history['val_rmse_metric'], label='Validation RMSE')
    plt.title('Model RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    # R²
    plt.subplot(2, 2, 4)
    plt.plot(history.history['r2_metric'], label='Training R²')
    plt.plot(history.history['val_r2_metric'], label='Validation R²')
    plt.title('Model R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./plots/model_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Evaluate model on test set
    logger.info("Evaluating model on test set...")
    metrics = evaluate_model(model, X_test, y_test, output_dir='./plots')
    
    # Quantize model for deployment
    logger.info("Quantizing model for deployment...")
    quantized_path = quantize_model(
        './models/biomass_cnn_lstm.h5',
        './models/biomass_cnn_lstm_quantized.tflite'
    )
    
    # Save evaluation metrics
    with open('./plots/evaluation_metrics.txt', 'w') as f:
        f.write(f"R² Score: {metrics['r2']:.4f}\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"MAE: {metrics['mae']:.4f}\n")
    
    logger.info("Model training completed!")
    logger.info(f"Model saved to: ./models/biomass_cnn_lstm.h5")
    logger.info(f"Quantized model saved to: {quantized_path}")
    logger.info(f"Training plots saved to: ./plots/")
    logger.info(f"Test R²: {metrics['r2']:.4f}")
    logger.info(f"Test RMSE: {metrics['rmse']:.4f}")
    logger.info(f"Test MAE: {metrics['mae']:.4f}")

if __name__ == "__main__":
    main()