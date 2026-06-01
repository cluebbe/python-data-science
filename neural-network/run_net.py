import numpy as np
from prepare_data import load_and_prepare_data
from neural_net import test
from train_net import train
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Network parameters
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

# Initialize weight matrices
w_input_hidden = np.random.uniform(-0.5, 0.5, (hidden_nodes, input_nodes))
w_hidden_output = np.random.uniform(-0.5, 0.5, (output_nodes, hidden_nodes))

# Load and prepare data
training_data, test_data, training_labels, test_labels = load_and_prepare_data("mnist_test.csv")

# Training loop
for i in range(len(training_data)):
    input_vector = training_data[i]
    label = int(training_labels[i])
    w_input_hidden, w_hidden_output = train(input_vector, label, w_input_hidden, w_hidden_output, learning_rate)

# Test the network on the first test example
first_test_input = test_data[0]
first_test_label = int(test_labels[0])
output_vector = test(first_test_input, w_input_hidden, w_hidden_output)

print("Output vector for the first test element:")
print(output_vector)
print("Predicted label:", np.argmax(output_vector))
print("True label:", first_test_label)

# Display the first test image
plt.imshow(first_test_input.reshape(28, 28), cmap='gray')
plt.title(f"True label: {first_test_label}")
plt.axis('off')
plt.show()
