# Import numpy for numerical operations
import numpy as np
# Import sigmoid activation function from neural_net
from neural_net import sigmoid

def train(input_vector, label, w_input_hidden, w_hidden_output, learning_rate):
    """
    Train the neural network on a single example.
    Args:
        input_vector (np.ndarray): Input vector of shape (784,)
        label (int): The correct label (0-9)
        w_input_hidden (np.ndarray): Weights from input to hidden layer
        w_hidden_output (np.ndarray): Weights from hidden to output layer
        learning_rate (float): Learning rate for weight updates
    Returns:
        tuple: Updated (w_input_hidden, w_hidden_output)
    """
    # Reshape input to column vector
    x = input_vector.reshape(-1, 1)
    # Calculate signals into hidden layer
    hidden_inputs = np.dot(w_input_hidden, x)
    # Apply sigmoid activation to hidden layer
    hidden_outputs = sigmoid(hidden_inputs)
    # Calculate signals into output layer
    final_inputs = np.dot(w_hidden_output, hidden_outputs)
    # Apply sigmoid activation to output layer
    final_outputs = sigmoid(final_inputs)
    # Create target vector (0.99 for correct label, 0.01 for others)
    targets = np.zeros((10, 1)) + 0.01
    targets[label] = 0.99
    # Calculate output layer error
    output_errors = targets - final_outputs
    # Calculate hidden layer error
    hidden_errors = np.dot(w_hidden_output.T, output_errors)
    # Update weights for hidden-output layer
    w_hidden_output += learning_rate * np.dot(
        (output_errors * final_outputs * (1.0 - final_outputs)),
        hidden_outputs.T
    )
    # Update weights for input-hidden layer
    w_input_hidden += learning_rate * np.dot(
        (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        x.T
    )
    # Return updated weights
    return w_input_hidden, w_hidden_output

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    # Network parameters
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.2
    # Initialize weight matrices with random values between -0.5 and 0.5
    w_input_hidden = np.random.uniform(
        -0.5, 0.5, (hidden_nodes, input_nodes)
    )
    w_hidden_output = np.random.uniform(
        -0.5, 0.5, (output_nodes, hidden_nodes)
    )
    # Create a random input vector and label for demonstration
    input_vector = np.random.rand(input_nodes)
    label = np.random.randint(0, output_nodes)
    # Display first 10 weights before training
    print("First 10 weights (input-hidden) before training:")
    print(w_input_hidden.flat[:10])
    print("First 10 weights (hidden-output) before training:")
    print(w_hidden_output.flat[:10])
    # Train the network once
    w_input_hidden, w_hidden_output = train(
        input_vector, label, w_input_hidden, w_hidden_output, learning_rate
    )
    # Display first 10 weights after training
    print("First 10 weights (input-hidden) after training:")
    print(w_input_hidden.flat[:10])
    print("First 10 weights (hidden-output) after training:")
    print(w_hidden_output.flat[:10])
