# Model Optimization for Above Ground Biomass Prediction
# Enhanced version with better hyperparameter tuning and quantization

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import os
import logging
import json
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import KerasTuner for hyperparameter optimization
try:
    from kerastuner.tuners import Hyperband, BayesianOptimization
    from kerastuner.engine.hyperparameters import HyperParameters
    KERAS_TUNER_AVAILABLE = True
    logger.info("KerasTuner is available")
except ImportError:
    logger.warning("KerasTuner not available. Install with: pip install keras-tuner")
    KERAS_TUNER_AVAILABLE = False

# Import the model building function from model_training
from model_training import build_cnn_lstm_model, evaluate_model

# 3.1 Enhanced Hyperparameter Tuning
def build_tunable_model(hp, input_shape):
    """
    Build model with hyperparameter tuning - fixed input shape
    """
    # Tunable hyperparameters
    num_filters_1 = hp.Int('num_filters_1', 16, 64, step=16)
    num_filters_2 = hp.Int('num_filters_2', 32, 128, step=32)
    num_filters_3 = hp.Int('num_filters_3', 64, 256, step=64)
    lstm_units = hp.Int('lstm_units', 64, 256, step=64)
    dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    l2_reg = hp.Float('l2_reg', 1e-5, 1e-3, sampling='log')
    
    # Optimizer choice
    optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    
    # Build model with tunable parameters
    model = build_cnn_lstm_model(
        input_shape,
        num_filters=(num_filters_1, num_filters_2, num_filters_3),
        lstm_units=lstm_units,
        dropout_rate=dropout_rate
    )
    
    # Configure optimizer
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:  # rmsprop
        optimizer = RMSprop(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return model

def tune_hyperparameters(sequences, targets, max_trials=20, executions_per_trial=3, tuner_type='hyperband'):
    """
    Enhanced hyperparameter tuning with multiple tuner options
    """
    if not KERAS_TUNER_AVAILABLE:
        logger.error("KerasTuner not available. Skipping hyperparameter tuning.")
        return None, None
    
    # Create tuner results directory
    os.makedirs('tuner_results', exist_ok=True)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, targets, test_size=0.2, random_state=42
    )
    
    # Get input shape
    input_shape = sequences.shape[1:]
    
    # Define callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    # Select tuner type
    if tuner_type == 'hyperband':
        tuner = Hyperband(
            lambda hp: build_tunable_model(hp, input_shape),
            objective='val_loss',
            max_epochs=50,
            factor=3,
            hyperband_iterations=2,
            directory='tuner_results',
            project_name='biomass_prediction_hyperband'
        )
    elif tuner_type == 'bayesian':
        tuner = BayesianOptimization(
            lambda hp: build_tunable_model(hp, input_shape),
            objective='val_loss',
            max_trials=max_trials,
            num_initial_points=5,
            directory='tuner_results',
            project_name='biomass_prediction_bayesian'
        )
    else:
        logger.error(f"Unknown tuner type: {tuner_type}")
        return None, None
    
    # Perform tuning
    logger.info(f"Starting hyperparameter search with {tuner_type} tuner...")
    start_time = time.time()
    
    tuner.search(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    search_time = time.time() - start_time
    logger.info(f"Hyperparameter search completed in {search_time:.2f} seconds")
    
    # Get best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Get best trial info
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    
    # Log results
    logger.info("\nBest Hyperparameters:")
    for param, value in best_hyperparameters.values.items():
        logger.info(f"{param}: {value}")
    
    logger.info(f"\nBest trial score: {best_trial.score}")
    logger.info(f"Best trial ID: {best_trial.trial_id}")
    
    # Save tuning results
    results = {
        'best_hyperparameters': best_hyperparameters.values,
        'best_score': best_trial.score,
        'search_time': search_time,
        'tuner_type': tuner_type
    }
    
    with open('tuner_results/best_hyperparameters.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_model, best_hyperparameters

# 3.2 Enhanced Model Quantization
def create_representative_dataset(sequences, num_samples=100):
    """
    Create representative dataset for quantization calibration
    """
    def representative_data_gen():
        # Sample random sequences from the dataset
        indices = np.random.choice(len(sequences), num_samples, replace=False)
        for i in indices:
            yield [sequences[i:i+1].astype(np.float32)]
    return representative_data_gen

def quantize_model(model_path, quantized_model_path, sequences=None, quantization_type='default'):
    """
    Enhanced model quantization with multiple options
    """
    try:
        # Load trained model
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded model from: {model_path}")
        
        # Convert to TensorFlow Lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantization_type == 'default':
            # Basic quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization_type == 'float16':
            # Float16 quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantization_type == 'int8':
            # Full integer quantization with calibration
            if sequences is None:
                raise ValueError("Sequences must be provided for int8 quantization")
            
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = create_representative_dataset(sequences)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        # Quantize model
        logger.info(f"Quantizing model with {quantization_type} quantization...")
        quantized_model = converter.convert()
        
        # Save quantized model
        os.makedirs(os.path.dirname(quantized_model_path), exist_ok=True)
        with open(quantized_model_path, 'wb') as f:
            f.write(quantized_model)
        
        # Get model sizes for comparison
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)  # MB
        compression_ratio = original_size / quantized_size
        
        logger.info(f"Original model size: {original_size:.2f} MB")
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        
        return quantized_model_path, compression_ratio
        
    except Exception as e:
        logger.error(f"Error during {quantization_type} quantization: {e}")
        return None, None

def evaluate_quantized_model(quantized_model_path, X_test, y_test):
    """
    Evaluate quantized model performance
    """
    try:
        # Load quantized model
        interpreter = tf.lite.Interpreter(model_path=quantized_model_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Make predictions
        predictions = []
        for i in range(len(X_test)):
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], X_test[i:i+1].astype(np.float32))
            
            # Run inference
            interpreter.invoke()
            
            # Get output tensor
            output = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output)
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        y_true_flat = y_test.flatten()
        y_pred_flat = predictions.flatten()
        
        # Remove NaN values
        mask = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
        y_true_flat = y_true_flat[mask]
        y_pred_flat = y_pred_flat[mask]
        
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        r2 = r2_score(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        
        logger.info(f"Quantized Model Performance:")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }
        
    except Exception as e:
        logger.error(f"Error evaluating quantized model: {e}")
        return None

# 3.3 Enhanced Optimization Execution
def main():
    """Main optimization execution function"""
    # Create necessary directories
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./optimization_results', exist_ok=True)
    
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
    
    # Check if base model exists
    base_model_path = './models/biomass_cnn_lstm.h5'
    if not os.path.exists(base_model_path):
        logger.error("Error: Base model not found. Please run model_training.py first.")
        return
    
    # Evaluate base model performance
    logger.info("Evaluating base model performance...")
    base_model = tf.keras.models.load_model(base_model_path)
    base_metrics = evaluate_model(base_model, X_test, y_test, './optimization_results/base_model')
    
    # Hyperparameter tuning
    if KERAS_TUNER_AVAILABLE:
        logger.info("Starting hyperparameter tuning...")
        best_model, best_hyperparameters = tune_hyperparameters(
            sequences, targets, max_trials=20, tuner_type='hyperband'
        )
        
        if best_model is not None:
            # Save best model
            best_model_path = './models/biomass_cnn_lstm_best.h5'
            best_model.save(best_model_path)
            logger.info(f"Best model saved to: {best_model_path}")
            
            # Evaluate best model
            logger.info("Evaluating best model performance...")
            best_metrics = evaluate_model(best_model, X_test, y_test, './optimization_results/best_model')
            
            # Use best model for quantization
            model_to_quantize = best_model_path
        else:
            model_to_quantize = base_model_path
    else:
        logger.warning("Skipping hyperparameter tuning. Using base model for quantization.")
        model_to_quantize = base_model_path
    
    # Quantize model with different methods
    quantization_results = {}
    
    # 1. Default quantization
    logger.info("\n1. Default Quantization:")
    default_quantized_path, default_compression = quantize_model(
        model_to_quantize,
        './models/biomass_cnn_lstm_default_quantized.tflite',
        quantization_type='default'
    )
    
    if default_quantized_path:
        default_metrics = evaluate_quantized_model(default_quantized_path, X_test, y_test)
        quantization_results['default'] = {
            'path': default_quantized_path,
            'compression': default_compression,
            'metrics': default_metrics
        }
    
    # 2. Float16 quantization
    logger.info("\n2. Float16 Quantization:")
    float16_quantized_path, float16_compression = quantize_model(
        model_to_quantize,
        './models/biomass_cnn_lstm_float16_quantized.tflite',
        quantization_type='float16'
    )
    
    if float16_quantized_path:
        float16_metrics = evaluate_quantized_model(float16_quantized_path, X_test, y_test)
        quantization_results['float16'] = {
            'path': float16_quantized_path,
            'compression': float16_compression,
            'metrics': float16_metrics
        }
    
    # 3. Int8 quantization with calibration
    logger.info("\n3. Int8 Quantization with Calibration:")
    int8_quantized_path, int8_compression = quantize_model(
        model_to_quantize,
        './models/biomass_cnn_lstm_int8_quantized.tflite',
        sequences=sequences,
        quantization_type='int8'
    )
    
    if int8_quantized_path:
        int8_metrics = evaluate_quantized_model(int8_quantized_path, X_test, y_test)
        quantization_results['int8'] = {
            'path': int8_quantized_path,
            'compression': int8_compression,
            'metrics': int8_metrics
        }
    
    # Compare all models
    logger.info("\n=== Model Comparison ===")
    logger.info(f"Base Model - R²: {base_metrics['r2']:.4f}, RMSE: {base_metrics['rmse']:.4f}")
    
    if 'best_metrics' in locals():
        logger.info(f"Best Model - R²: {best_metrics['r2']:.4f}, RMSE: {best_metrics['rmse']:.4f}")
    
    for quant_type, results in quantization_results.items():
        if results['metrics']:
            logger.info(f"{quant_type.upper()} Quantized - R²: {results['metrics']['r2']:.4f}, "
                       f"RMSE: {results['metrics']['rmse']:.4f}, Compression: {results['compression']:.2f}x")
    
    # Save optimization results
    optimization_summary = {
        'base_model': base_metrics,
        'quantization_results': quantization_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if 'best_metrics' in locals():
        optimization_summary['best_model'] = best_metrics
    
    with open('./optimization_results/optimization_summary.json', 'w') as f:
        json.dump(optimization_summary, f, indent=2)
    
    logger.info("\nModel optimization completed!")
    logger.info("Available models:")
    logger.info(f"- Original: {base_model_path}")
    if 'best_model_path' in locals():
        logger.info(f"- Optimized: {best_model_path}")
    for quant_type, results in quantization_results.items():
        logger.info(f"- {quant_type.upper()} Quantized: {results['path']}")

if __name__ == "__main__":
    main()