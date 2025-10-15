"""
CNN+LSTM Biomass Prediction Model - Academic Research Implementation
Deep Learning Architecture for Above Ground Biomass Estimation

This module demonstrates the machine learning approach for biomass prediction
using Convolutional Neural Networks (CNN) for spatial feature extraction
and Long Short-Term Memory (LSTM) networks for temporal pattern analysis.

Author: Research Team
Purpose: Academic demonstration of deep learning for biomass prediction
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class BiomassPredictor:
    """
    CNN+LSTM model for biomass prediction using satellite time series data.
    
    Architecture Components:
    - CNN layers: Extract spatial features from satellite imagery
    - LSTM layers: Capture temporal patterns in vegetation indices
    - Dense layers: Final biomass classification/regression
    """
    
    def __init__(self, input_shape=(64, 64, 12, 7), num_classes=2):
        """
        Initialize the biomass prediction model.
        
        Args:
            input_shape: (height, width, time_steps, channels)
                        Represents satellite image patches over time
            num_classes: Number of biomass categories (Low=0, High=1)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.training_history = None
        
        logger.info("Biomass Predictor initialized")
        logger.info(f"Input shape: {input_shape}")
        logger.info(f"Output classes: {num_classes}")
    
    def build_cnn_lstm_architecture(self):
        """
        Build the CNN+LSTM architecture for biomass prediction.
        
        The model processes satellite time series data through:
        1. Spatial feature extraction (CNN)
        2. Temporal pattern recognition (LSTM)
        3. Classification/regression (Dense layers)
        """
        logger.info("Building CNN+LSTM model architecture...")
        
        # Input layer for satellite time series
        inputs = keras.Input(shape=self.input_shape, name='satellite_timeseries')
        
        # Spatial Feature Extraction with CNN
        # TimeDistributed applies CNN to each time step
        x = layers.TimeDistributed(
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            name='spatial_conv1'
        )(inputs)
        
        x = layers.TimeDistributed(
            layers.BatchNormalization(),
            name='spatial_bn1'
        )(x)
        
        x = layers.TimeDistributed(
            layers.MaxPooling2D((2, 2)),
            name='spatial_pool1'
        )(x)
        
        # Second convolutional block
        x = layers.TimeDistributed(
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            name='spatial_conv2'
        )(x)
        
        x = layers.TimeDistributed(
            layers.BatchNormalization(),
            name='spatial_bn2'
        )(x)
        
        x = layers.TimeDistributed(
            layers.MaxPooling2D((2, 2)),
            name='spatial_pool2'
        )(x)
        
        # Global average pooling to reduce spatial dimensions
        x = layers.TimeDistributed(
            layers.GlobalAveragePooling2D(),
            name='spatial_gap'
        )(x)
        
        # Temporal Pattern Recognition with LSTM
        x = layers.LSTM(64, return_sequences=True, name='temporal_lstm1')(x)
        x = layers.Dropout(0.3, name='temporal_dropout1')(x)
        
        x = layers.LSTM(32, return_sequences=False, name='temporal_lstm2')(x)
        x = layers.Dropout(0.3, name='temporal_dropout2')(x)
        
        # Classification layers
        x = layers.Dense(64, activation='relu', name='classification_dense')(x)
        x = layers.BatchNormalization(name='classification_bn')(x)
        x = layers.Dropout(0.4, name='classification_dropout')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='biomass_output'
        )(x)
        
        # Create and compile model
        self.model = keras.Model(
            inputs=inputs, 
            outputs=outputs, 
            name='BiomassPredictor'
        )
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model architecture built successfully")
        logger.info(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def prepare_satellite_data(self, num_samples=1000):
        """
        Prepare synthetic satellite data for model development and testing.
        
        In a real implementation, this would load actual satellite imagery
        from sources like MODIS, Sentinel-2, or Landsat.
        
        Args:
            num_samples: Number of sample patches to generate
            
        Returns:
            X: Satellite time series data
            y: Biomass class labels
        """
        logger.info(f"Preparing {num_samples} satellite data samples...")
        
        # Generate synthetic satellite time series data
        # In practice, this would be real satellite imagery
        np.random.seed(42)  # For reproducible results
        
        # Create satellite data with realistic characteristics
        X = np.random.randn(num_samples, *self.input_shape).astype(np.float32)
        
        # Simulate realistic satellite data ranges
        # Channel 0: NDVI (-1 to 1)
        X[:, :, :, :, 0] = np.tanh(X[:, :, :, :, 0])
        
        # Channel 1: EVI (-1 to 1)  
        X[:, :, :, :, 1] = np.tanh(X[:, :, :, :, 1] * 0.8)
        
        # Channel 2: LAI (0 to 8)
        X[:, :, :, :, 2] = np.abs(X[:, :, :, :, 2]) * 2
        
        # Channel 3: Temperature (0 to 40°C)
        X[:, :, :, :, 3] = (X[:, :, :, :, 3] + 2) * 10
        
        # Channels 4-6: Other vegetation indices
        X[:, :, :, :, 4:] = np.abs(X[:, :, :, :, 4:])
        
        # Generate labels based on vegetation health
        # High NDVI and LAI typically indicate high biomass
        mean_ndvi = np.mean(X[:, :, :, :, 0], axis=(1, 2, 3))
        mean_lai = np.mean(X[:, :, :, :, 2], axis=(1, 2, 3))
        
        # Simple biomass classification rule
        biomass_indicator = (mean_ndvi + mean_lai / 4) / 2
        y = (biomass_indicator > 0.2).astype(int)
        
        logger.info(f"Data prepared: {X.shape}, Labels: {y.shape}")
        logger.info(f"Class distribution: Low={np.sum(y==0)}, High={np.sum(y==1)}")
        
        return X, y
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=20):
        """
        Train the CNN+LSTM model on satellite data.
        
        Args:
            X_train, y_train: Training data and labels
            X_val, y_val: Validation data and labels
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        logger.info("Starting model training...")
        
        # Define callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = history.history
        logger.info("Model training completed")
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test, y_test: Test data and labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        # Create evaluation report
        evaluation_results = {
            'accuracy': float(accuracy),
            'class_0': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1_score': float(f1[0]),
                'support': int(support[0])
            },
            'class_1': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1_score': float(f1[1]),
                'support': int(support[1])
            }
        }
        
        logger.info("Model evaluation completed")
        return evaluation_results, y_pred
    
    def plot_training_curves(self):
        """
        Plot training and validation curves.
        """
        if not self.training_history:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.training_history['loss'], label='Training Loss')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(self.training_history['accuracy'], label='Training Accuracy')
        ax2.plot(self.training_history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("./outputs/model_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info("Training curves plotted and saved")
    
    def visualize_model_architecture(self):
        """
        Visualize the model architecture.
        """
        if self.model is None:
            logger.warning("Model not built yet")
            return
        
        # Print model summary
        print("\nModel Architecture Summary:")
        print("=" * 50)
        self.model.summary()
        
        # Try to plot model architecture (if graphviz is available)
        try:
            keras.utils.plot_model(
                self.model,
                to_file="./outputs/model_analysis/model_architecture.png",
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB'
            )
            logger.info("Model architecture diagram saved")
        except Exception as e:
            logger.warning(f"Could not create architecture diagram: {e}")
    
    def predict_biomass(self, satellite_data):
        """
        Predict biomass class for new satellite data.
        
        Args:
            satellite_data: Input satellite time series data
            
        Returns:
            Predicted biomass class and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get prediction probabilities
        predictions = self.model.predict(satellite_data)
        
        # Get class predictions and confidence
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        return predicted_classes, confidence_scores

def demonstrate_biomass_prediction():
    """
    Demonstrate the complete biomass prediction workflow.
    """
    print("CNN+LSTM Biomass Prediction Model Demonstration")
    print("=" * 60)
    
    # Initialize model
    predictor = BiomassPredictor()
    
    # Build architecture
    model = predictor.build_cnn_lstm_architecture()
    
    # Show model architecture
    predictor.visualize_model_architecture()
    
    # Prepare data
    X, y = predictor.prepare_satellite_data(num_samples=500)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"\nDataset splits:")
    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples") 
    print(f"Testing: {len(X_test)} samples")
    
    # Train model
    print("\nTraining model...")
    history = predictor.train_model(X_train, y_train, X_val, y_val, epochs=10)
    
    # Evaluate model
    print("\nEvaluating model...")
    results, y_pred = predictor.evaluate_model(X_test, y_test)
    
    # Print results
    print(f"\nModel Performance:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Low Biomass - Precision: {results['class_0']['precision']:.3f}, "
          f"Recall: {results['class_0']['recall']:.3f}")
    print(f"High Biomass - Precision: {results['class_1']['precision']:.3f}, "
          f"Recall: {results['class_1']['recall']:.3f}")
    
    # Plot training curves
    predictor.plot_training_curves()
    
    print("\nDemonstration completed successfully!")
    return predictor, results

if __name__ == "__main__":
    """
    Run the biomass prediction model demonstration.
    """
    predictor, results = demonstrate_biomass_prediction()