"""
Deep Learning CNN approach for audio forgery detection
Expected performance: 80-90% accuracy with proper spectrograms
"""
import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

def audio_to_enhanced_spectrogram(audio_path, target_shape=(128, 128)):
    """Convert audio to enhanced spectrogram for CNN training"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=target_shape[0], 
            hop_length=512,
            n_fft=2048,
            fmin=20,
            fmax=8000
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # Resize to target shape
        if mel_spec_norm.shape[1] != target_shape[1]:
            # Interpolate to target width
            from scipy import ndimage
            mel_spec_norm = ndimage.zoom(
                mel_spec_norm, 
                (1, target_shape[1] / mel_spec_norm.shape[1]), 
                order=1
            )
        
        return mel_spec_norm
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return np.zeros(target_shape)

def create_cnn_model(input_shape=(128, 128, 1), num_classes=2):
    """Create CNN model optimized for audio forgery detection"""
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth conv block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Global average pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def prepare_cnn_data(dataset_root, target_shape=(128, 128)):
    """Prepare spectrogram data for CNN training"""
    
    print("🔧 PREPARING CNN DATA FROM SPECTROGRAMS")
    print("=" * 60)
    
    # Load labels
    labels_path = os.path.join(dataset_root, 'metadata', 'labels.csv')
    labels_df = pd.read_csv(labels_path)
    
    print(f"📊 Total files: {len(labels_df)}")
    
    # Extract spectrograms
    spectrograms = []
    labels = []
    splits = []
    
    for idx, row in labels_df.iterrows():
        audio_path = os.path.join(dataset_root, row['audio_path'])
        
        if os.path.exists(audio_path):
            # Convert to spectrogram
            spec = audio_to_enhanced_spectrogram(audio_path, target_shape)
            spectrograms.append(spec)
            labels.append(1 if row['label'] == 'forged' else 0)
            splits.append(row['split'])
        
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(labels_df)} files...")
    
    # Convert to numpy arrays
    X = np.array(spectrograms)
    y = np.array(labels)
    splits = np.array(splits)
    
    # Add channel dimension for CNN
    X = X[..., np.newaxis]
    
    print(f"\n📊 Data preparation complete:")
    print(f"Spectrograms shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    train_mask = splits == 'train'
    val_mask = splits == 'val'
    test_mask = splits == 'test'
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\n📊 Data splits:")
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def train_cnn_model(dataset_root, epochs=50, batch_size=16):
    """Train CNN model on aggressive forgery dataset"""
    
    print("🚀 TRAINING CNN MODEL FOR AUDIO FORGERY DETECTION")
    print("=" * 60)
    
    # Prepare data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_cnn_data(dataset_root)
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, 2)
    y_val_cat = keras.utils.to_categorical(y_val, 2)
    y_test_cat = keras.utils.to_categorical(y_test, 2)
    
    # Create model
    input_shape = X_train.shape[1:]
    model = create_cnn_model(input_shape)
    
    print(f"\n🏗️ Model architecture:")
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Train model
    print(f"\n🏋️ Training model for {epochs} epochs...")
    history = model.fit(
        X_train, y_train_cat,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\n📊 EVALUATING MODEL ON TEST SET:")
    print("=" * 60)
    
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Detailed predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nDetailed Results:")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Original', 'Forged']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    # Prediction distribution
    pred_dist = np.bincount(y_pred, minlength=2)
    print(f"\nPrediction Distribution:")
    print(f"Predicted Original: {pred_dist[0]}")
    print(f"Predicted Forged: {pred_dist[1]}")
    
    # Check for bias
    if pred_dist[0] == 0:
        print(f"⚠️ WARNING: Model always predicts FORGED")
    elif pred_dist[1] == 0:
        print(f"⚠️ WARNING: Model always predicts ORIGINAL")
    else:
        print(f"✅ Model shows balanced predictions")
    
    # Performance assessment
    print(f"\n📈 PERFORMANCE ASSESSMENT:")
    if f1 > 0.85:
        print(f"🎉 EXCELLENT: F1 = {f1:.4f} - Deep learning working well!")
    elif f1 > 0.75:
        print(f"✅ GOOD: F1 = {f1:.4f} - Significant improvement over traditional ML")
    elif f1 > 0.65:
        print(f"🔶 FAIR: F1 = {f1:.4f} - Better than traditional ML but room for improvement")
    else:
        print(f"❌ POOR: F1 = {f1:.4f} - May need better data or different approach")
    
    # Save model
    model_path = "cnn_forgery_detector.h5"
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    return model, history, f1

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cnn_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    dataset_root = "aggressive_forgery_test_dataset"
    
    if os.path.exists(dataset_root):
        print("🎯 Starting CNN training on aggressive forgery dataset...")
        model, history, f1_score = train_cnn_model(
            dataset_root, 
            epochs=30,  # Reduced for faster testing
            batch_size=8   # Smaller batch size for limited data
        )
        
        # Plot training history
        plot_training_history(history)
        
        print(f"\n🏆 FINAL CNN RESULTS:")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Expected improvement: {f1_score/0.62:.2f}x over traditional ML")
        
    else:
        print(f"❌ Dataset not found: {dataset_root}")
        print("Please run regenerate_aggressive_dataset.py first")
