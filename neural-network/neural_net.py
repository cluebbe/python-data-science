# Import numpy for numerical operations
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    """
    Compute the sigmoid activation for the input x.
    """
    return 1 / (1 + np.exp(-x))

# Forward pass through the neural network
def test(input_vector, w_input_hidden, w_hidden_output):
    """
    Propagate input through the network.
    Args:
        input_vector (np.ndarray): Input vector
        w_input_hidden (np.ndarray): Weights from input to hidden layer
        w_hidden_output (np.ndarray): Weights from hidden to output layer
    Returns:
        np.ndarray: Output vector from the network
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
    # Return as 1D numpy array
    return final_outputs.flatten()

if __name__ == "__main__":
    # Import the data preparation function
    from prepare_data import load_and_prepare_data
    # Set random seed for reproducibility
    np.random.seed(40)
    # Network parameters
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    # Initialize weight matrices with random values between -0.5 and 0.5
    w_input_hidden = np.random.uniform(
        -0.5, 0.5, (hidden_nodes, input_nodes)
    )
    w_hidden_output = np.random.uniform(
        -0.5, 0.5, (output_nodes, hidden_nodes)
    )
    # Load and prepare data
    _, test_data, _, test_labels = load_and_prepare_data("mnist_test.csv")
    # Test the network on the first test example
    first_test_input = test_data[0]
    first_test_label = int(test_labels[0])
    output_vector = test(first_test_input, w_input_hidden, w_hidden_output)
    print("Output vector for the first test element:")
    print(output_vector)
    print("Predicted label:", np.argmax(output_vector))
    print("True label:", first_test_label)
