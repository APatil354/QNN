import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

def build_unet(input_shape):
    """ Builds a U-Net model suitable for small image dimensions. """
    inputs = Input(input_shape)

    # Encoder (Contracting Path)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1) # Size -> 16x16

    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2) # Size -> 8x8

    # Bottleneck
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)

    # Decoder (Expansive Path)
    u4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c3) # Size -> 16x16
    u4 = Concatenate()([u4, c2]) # Skip connection
    c4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u4)
    c4 = Dropout(0.1)(c4)
    c4 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)

    u5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c4) # Size -> 32x32
    u5 = Concatenate()([u5, c1]) # Skip connection
    c5 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Output layer
    outputs = Conv2D(2, (1, 1), activation='linear')(c5) # 2 channels for Dx, Dy

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def plot_training_history(history):
    """ Plots the training and validation loss over epochs. """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Epoch vs. Error (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (on Scaled Data)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(model, x_test, y_test_scaled, scaler, grid_extent, num_samples=2):
    """ Plots un-scaled actual vs. predicted results with BC overlay. """
    if len(x_test) < num_samples:
        num_samples = len(x_test)
        print(f"Warning: Only {num_samples} test samples available to plot.")

    predictions_scaled = model.predict(x_test)

    h, w, c = predictions_scaled.shape[1], predictions_scaled.shape[2], predictions_scaled.shape[3]
    predictions_actual = scaler.inverse_transform(predictions_scaled.reshape(-1, c)).reshape(-1, h, w, c)
    y_test_actual = scaler.inverse_transform(y_test_scaled.reshape(-1, c)).reshape(-1, h, w, c)
    
    for i in range(num_samples):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Sample {i+1}: Actual vs. Predicted Displacement (Original Units)', fontsize=16)
        
        bc_map = x_test[i, :, :, 1]
        bc_mask = np.ma.masked_where(bc_map == 0, bc_map)

        combined_dx = np.concatenate([y_test_actual[i, :, :, 0].ravel(), predictions_actual[i, :, :, 0].ravel()])
        dx_min, dx_max = np.percentile(combined_dx, [2, 98])
        combined_dy = np.concatenate([y_test_actual[i, :, :, 1].ravel(), predictions_actual[i, :, :, 1].ravel()])
        dy_min, dy_max = np.percentile(combined_dy, [2, 98])

        ax = axes[0, 0]
        im = ax.imshow(y_test_actual[i, :, :, 0], cmap='RdYlGn_r', vmin=dx_min, vmax=dx_max, origin='lower', extent=grid_extent)
        ax.set_title('Actual Displacement Dx'); fig.colorbar(im, ax=ax)

        ax = axes[0, 1]
        im = ax.imshow(predictions_actual[i, :, :, 0], cmap='RdYlGn_r', vmin=dx_min, vmax=dx_max, origin='lower', extent=grid_extent)
        ax.imshow(bc_mask, cmap='binary', origin='lower', extent=grid_extent, alpha=0.5) 
        ax.set_title('Predicted Displacement Dx (with BCs)'); fig.colorbar(im, ax=ax)

        ax = axes[1, 0]
        im = ax.imshow(y_test_actual[i, :, :, 1], cmap='RdYlGn_r', vmin=dy_min, vmax=dy_max, origin='lower', extent=grid_extent)
        ax.set_title('Actual Displacement Dy'); fig.colorbar(im, ax=ax)

        ax = axes[1, 1]
        im = ax.imshow(predictions_actual[i, :, :, 1], cmap='RdYlGn_r', vmin=dy_min, vmax=dy_max, origin='lower', extent=grid_extent)
        ax.imshow(bc_mask, cmap='binary', origin='lower', extent=grid_extent, alpha=0.5)
        ax.set_title('Predicted Displacement Dy (with BCs)'); fig.colorbar(im, ax=ax)

        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

def plot_error_maps(model, x_test, y_test_scaled, scaler, grid_extent, num_samples=2):
    """ Calculates and plots the un-scaled error with BC overlay. """
    if len(x_test) < num_samples:
        num_samples = len(x_test)

    predictions_scaled = model.predict(x_test)
    h, w, c = predictions_scaled.shape[1], predictions_scaled.shape[2], predictions_scaled.shape[3]
    predictions_actual = scaler.inverse_transform(predictions_scaled.reshape(-1, c)).reshape(-1, h, w, c)
    y_test_actual = scaler.inverse_transform(y_test_scaled.reshape(-1, c)).reshape(-1, h, w, c)
    error = y_test_actual - predictions_actual

    for i in range(num_samples):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Sample {i+1}: Prediction Error (Actual - Predicted)', fontsize=16)

        bc_map = x_test[i, :, :, 1]
        bc_mask = np.ma.masked_where(bc_map == 0, bc_map)

        error_dx = error[i, :, :, 0]
        vmax_dx = np.max(np.abs(error_dx)) if np.max(np.abs(error_dx)) > 1e-9 else 1.0
        ax = axes[0]
        im = ax.imshow(error_dx, cmap='coolwarm', vmin=-vmax_dx, vmax=vmax_dx, origin='lower', extent=grid_extent)
        ax.imshow(bc_mask, cmap='binary', origin='lower', extent=grid_extent, alpha=0.5)
        ax.set_title('Dx Error (with BCs)'); fig.colorbar(im, ax=ax)

        error_dy = error[i, :, :, 1]
        vmax_dy = np.max(np.abs(error_dy)) if np.max(np.abs(error_dy)) > 1e-9 else 1.0
        ax = axes[1]
        im = ax.imshow(error_dy, cmap='coolwarm', vmin=-vmax_dy, vmax=vmax_dy, origin='lower', extent=grid_extent)
        ax.imshow(bc_mask, cmap='binary', origin='lower', extent=grid_extent, alpha=0.5)
        ax.set_title('Dy Error (with BCs)'); fig.colorbar(im, ax=ax)

        plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()

def evaluate_model_performance(model, x_test, y_test_scaled, scaler):
    """ Calculates and prints performance metrics on UN-SCALED data. """
    print("\n--- Advanced Model Performance Evaluation (Un-scaled Data) ---")

    predictions_scaled = model.predict(x_test)
    
    n_samples, h, w, c = y_test_scaled.shape
    y_pred_actual = scaler.inverse_transform(predictions_scaled.reshape(-1, c)).reshape(n_samples, h, w, c)
    y_test_actual = scaler.inverse_transform(y_test_scaled.reshape(-1, c)).reshape(n_samples, h, w, c)

    y_true_flat = y_test_actual.flatten()
    y_pred_flat = y_pred_actual.flatten()
    
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    rmse = np.sqrt(np.mean(np.square(y_true_flat - y_pred_flat)))
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    r2 = r2_score(y_true_flat, y_pred_flat)
    print(f"R-squared (R²) Score: {r2:.4f}")
    print("-----------------------------------------------------------\n")


if __name__ == "__main__":
    # 1. Load Data
    print("Loading data...")
    try:
        # --- ADJUSTED: Use the filenames from the generation script ---
        X = np.load("X_data_with_props.npy")
        Y = np.load("Y_data_with_props.npy")
    except FileNotFoundError:
        print("Error: Make sure 'X_data_with_props.npy' and 'Y_data_with_props.npy' are in the directory.")
        exit()

    print(f"Data loaded. Total samples: {len(X)}. X shape: {X.shape}, Y shape: {Y.shape}")
    
    # Scale Y Data
    n_samples, h, w, c = Y.shape
    Y_reshaped = Y.reshape(-1, c)
    scaler = StandardScaler()
    Y_scaled_reshaped = scaler.fit_transform(Y_reshaped)
    Y_scaled = Y_scaled_reshaped.reshape(n_samples, h, w, c)
    print("Y data has been normalized using StandardScaler.")

    # Split Data
    X_train, X_test, Y_train_scaled, Y_test_scaled = train_test_split(X, Y_scaled, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # Build and Compile Model
    input_shape = X_train.shape[1:]
    # --- CHANGED: Call the new U-Net builder ---
    model = build_unet(input_shape)
    optimizer = Adam(learning_rate=1e-4) # A learning rate of 1e-4 is often a good start for U-Nets
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    print("\n## Model Summary")
    model.summary()

    # Define Callbacks and Train
    print("\nStarting model training...")
    early_stopper = EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)
    
    history = model.fit(
        X_train, Y_train_scaled,
        validation_data=(X_test, Y_test_scaled),
        epochs=200,       # U-Nets can sometimes train faster
        batch_size=4,       # Smaller batch size for potentially larger model
        verbose=1,
        callbacks=[early_stopper]
    )
    print("Training finished.")
    
    evaluate_model_performance(model, X_test, Y_test_scaled, scaler)
    GRID_TOTAL_WIDTH = 32.0 
    grid_extent = [-GRID_TOTAL_WIDTH/2, GRID_TOTAL_WIDTH/2, -GRID_TOTAL_WIDTH/2, GRID_TOTAL_WIDTH/2]

    plot_training_history(history)
    plot_predictions(model, X_test, Y_test_scaled, scaler, grid_extent, num_samples=3)
    print("Generating error maps (Actual - Predicted)...")
    plot_error_maps(model, X_test, Y_test_scaled, scaler, grid_extent, num_samples=3)