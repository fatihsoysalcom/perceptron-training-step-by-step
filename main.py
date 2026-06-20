import math

# --- Activation Function ---
def sigmoid(x):
    """The sigmoid activation function."""
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

# --- Hyperparameters ---
learning_rate = 0.1
epochs = 100 # Increased epochs for better learning demonstration

# --- Dataset (Simple AND gate logic for a single neuron) ---
# Inputs: [input1, input2]
# Targets: [output]
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 0, 0, 1] # Target outputs for AND gate

# --- Initialize Weights and Bias ---
# Using fixed values for reproducibility in a simple example
weights = [0.5, -0.5] # Initial weights for input1 and input2
bias = 0.1 # Initial bias

print("--- Initial Setup ---")
print(f"Initial Weights: {weights}")
print(f"Initial Bias: {bias}")
print(f"Learning Rate: {learning_rate}")
print("-" * 30)

# --- Training Loop ---
for epoch in range(epochs):
    total_loss = 0
    if (epoch + 1) % 10 == 0 or epoch == 0: # Print detailed steps every 10 epochs or first epoch
        print(f"\n--- Epoch {epoch + 1}/{epochs} (Detailed) ---")
    else:
        # Print summary for other epochs to keep output concise
        for i in range(len(X)):
            x_input = X[i]
            y_true = Y[i]
            weighted_sum = sum(x_input[j] * weights[j] for j in range(len(x_input))) + bias
            y_pred = sigmoid(weighted_sum)
            total_loss += (y_true - y_pred)**2
        avg_epoch_loss = total_loss / len(X)
        print(f"Epoch {epoch + 1}/{epochs} Average Loss: {avg_epoch_loss:.4f}")
        continue # Skip detailed printing for this epoch

    for i in range(len(X)):
        x_input = X[i]
        y_true = Y[i]

        print(f"\n  Input: {x_input}, Target: {y_true}")

        # --- 1. Forward Propagation (İleri Besleme) ---
        # Calculate weighted sum (z = w1*x1 + w2*x2 + b)
        weighted_sum = 0
        for j in range(len(x_input)):
            weighted_sum += x_input[j] * weights[j]
        weighted_sum += bias
        # Apply activation function to get prediction (y_pred = sigmoid(z))
        y_pred = sigmoid(weighted_sum)

        print(f"    [Forward Pass] Weighted Sum (z): {weighted_sum:.4f}")
        print(f"    [Forward Pass] Predicted Output (y_pred): {y_pred:.4f}")

        # --- 2. Loss Calculation (Kayıp Fonksiyonu) ---
        # Using Mean Squared Error (MSE) loss
        loss = (y_true - y_pred)**2
        total_loss += loss
        print(f"    [Loss] Mean Squared Error: {loss:.4f}")

        # --- 3. Backpropagation (Geri Yayılım) ---
        # Calculate gradients for weights and bias

        # Derivative of loss with respect to y_pred (dLoss/dY_pred)
        d_loss_d_y_pred = -2 * (y_true - y_pred)
        # Derivative of y_pred with respect to weighted_sum (dY_pred/dZ)
        # This is the derivative of the sigmoid function at 'z'
        d_y_pred_d_z = sigmoid_derivative(weighted_sum)

        # Gradient for weights (dLoss/dWi = dLoss/dY_pred * dY_pred/dZ * dZ/dWi)
        # dZ/dWi is simply Xi
        d_loss_d_weights = [0] * len(weights)
        for j in range(len(weights)):
            d_loss_d_weights[j] = d_loss_d_y_pred * d_y_pred_d_z * x_input[j]

        # Gradient for bias (dLoss/dBias = dLoss/dY_pred * dY_pred/dZ * dZ/dBias)
        # dZ/dBias is 1
        d_loss_d_bias = d_loss_d_y_pred * d_y_pred_d_z * 1 

        print(f"    [Backprop] dLoss/dY_pred: {d_loss_d_y_pred:.4f}")
        print(f"    [Backprop] dY_pred/dZ: {d_y_pred_d_z:.4f}")
        print(f"    [Backprop] Gradients for Weights: {[f'{g:.4f}' for g in d_loss_d_weights]}")
        print(f"    [Backprop] Gradient for Bias: {d_loss_d_bias:.4f}")

        # --- Update Weights and Bias ---
        # Apply gradient descent to update weights and bias
        for j in range(len(weights)):
            weights[j] -= learning_rate * d_loss_d_weights[j]
        bias -= learning_rate * d_loss_d_bias

        print(f"    [Update] New Weights: {[f'{w:.4f}' for w in weights]}")
        print(f"    [Update] New Bias: {bias:.4f}")

    avg_epoch_loss = total_loss / len(X)
    print(f"\nEpoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")
    print("-" * 30)

print("\n--- Training Complete ---")
print(f"Final Weights: {[f'{w:.4f}' for w in weights]}")
print(f"Final Bias: {bias:.4f}")

# --- Test final model ---
print("\n--- Testing Final Model ---")
for i in range(len(X)):
    x_input = X[i]
    y_true = Y[i]
    weighted_sum = sum(x_input[j] * weights[j] for j in range(len(x_input))) + bias
    y_pred = sigmoid(weighted_sum)
    # Round prediction to nearest integer for binary classification interpretation
    predicted_class = round(y_pred)
    print(f"  Input: {x_input}, Actual: {y_true}, Predicted: {y_pred:.4f} (Class: {predicted_class})")
