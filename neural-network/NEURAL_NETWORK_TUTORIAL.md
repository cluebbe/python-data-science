# Neural Networks from Scratch — Step-by-Step Tutorial

## Introduction to Neural Networks

A **neural network** is a supervised machine learning algorithm loosely inspired by the structure of biological brains. It consists of layers of interconnected nodes (neurons) that transform an input signal step by step until a final prediction is produced.

### How it works

Each neuron receives a weighted sum of its inputs, passes it through a non-linear **activation function**, and forwards the result to the next layer. By stacking multiple layers, the network can learn complex, non-linear patterns that simpler models cannot.

The network has three types of layers:

| Layer | Role |
|---|---|
| Input layer | Receives the raw features (one node per feature) |
| Hidden layer(s) | Learns intermediate representations |
| Output layer | Produces the final prediction (one node per class) |

### Forward pass

For each layer, the computation is:

```
hidden_inputs  = W_input_hidden  · x          (matrix multiply)
hidden_outputs = sigmoid(hidden_inputs)        (activation)
final_inputs   = W_hidden_output · hidden_outputs
final_outputs  = sigmoid(final_inputs)
```

The **sigmoid** function squashes any value into (0, 1), making it interpretable as a probability:

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

### Backpropagation and weight updates

After a forward pass, the network computes the **error** — the difference between the target and the actual output. This error is propagated backwards through the layers (backpropagation) to compute how much each weight contributed to the mistake. Weights are then nudged in the direction that reduces the error:

```
ΔW = learning_rate · (error · sigmoid_derivative) · previous_layer_outputs
```

The sigmoid derivative is: `σ(z) · (1 − σ(z))`, which is efficient to compute since `σ(z)` was already calculated in the forward pass.

### The dataset — MNIST

This tutorial uses a subset of the **MNIST** dataset — 10,000 handwritten digit images (0–9), each stored as a 28×28 pixel grid flattened into a vector of 784 values. The goal is to train a network that correctly classifies which digit each image shows.

### Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Can learn complex, non-linear patterns | Requires more data than simpler models |
| Scalable to large datasets | Computationally expensive to train |
| Flexible architecture | Harder to interpret than decision trees |
| Strong performance on image and text data | Sensitive to hyperparameter choices |

---

## Preparation — Environment Setup

Before running any code, install Python and set up an isolated environment.

**Install Python 3.9 or newer** from [python.org](https://www.python.org/downloads/). Verify it is available in your terminal:

```bash
python3 --version
```

> **Windows users:** during installation, tick **"Add Python to PATH"** so the `python` and `pip` commands are available in your terminal.

Then set up an isolated environment:

```bash
# 1. Create and enter your project folder
mkdir neural-network && cd neural-network

# 2. Create a virtual environment (run once)
python3 -m venv venv

# 3. Activate it
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install numpy pandas matplotlib scikit-learn

# 5. When you're done, deactivate
deactivate
```

> **Why a virtual environment?** It keeps the packages for this project separate from your system Python and other projects, avoiding version conflicts.

---

## Preparation — Imports

Each file in this tutorial uses a focused set of imports:

```python
# prepare_data.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# show_data.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# neural_net.py
import numpy as np

# train_net.py
import numpy as np
from neural_net import sigmoid

# run_net.py
import numpy as np
import matplotlib.pyplot as plt
from prepare_data import load_and_prepare_data
from neural_net import test
from train_net import train
```

- **numpy** — numerical operations and matrix multiplications
- **pandas** — reading the CSV data file
- **matplotlib** — visualising digit images and training results
- **sklearn.model_selection** — splitting data into train/test sets

---

## Step 1 — Load & Prepare the Data (`prepare_data.py`)

Read the MNIST CSV file, split it into training and test sets, and normalise the pixel values. The CSV has no header — the first column is the label (0–9) and the remaining 784 columns are pixel values (0–255).

Normalise pixel values to the range **[0.01, 1.0]** rather than [0, 1]. This avoids feeding a zero input into the network (a zero input produces zero gradient, stalling weight updates) while keeping all values within the sigmoid's sensitive range.

<details>
<summary>Solution</summary>

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data(path):
    # Read the CSV file into a pandas DataFrame without headers
    df = pd.read_csv(path, header=None)
    datalist = df.values.tolist()

    # Separate labels (first column) from pixel data (remaining columns)
    label_list = [row[0] for row in datalist]
    data_list  = [row[1:] for row in datalist]

    # Split into 80% training, 20% test
    training_data, test_data, training_labels, test_labels = train_test_split(
        np.array(data_list),
        np.array(label_list),
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    # Normalise pixel values from [0, 255] to [0.01, 1.0]
    training_data = np.asarray(training_data) / 255 * 0.99 + 0.01
    test_data     = np.asarray(test_data)     / 255 * 0.99 + 0.01

    return training_data, test_data, training_labels, test_labels
```

**Why [0.01, 1.0] and not [0, 1]?**
The sigmoid function saturates (outputs very close to 0 or 1) at extreme inputs. A pixel value of 0 would produce `sigmoid(w · 0) = sigmoid(0) = 0.5` regardless of the weight — no information is passed. Shifting the range to [0.01, 1.0] ensures even "dark" pixels carry a small signal.

**Why shuffle before splitting?**
The CSV may be ordered by digit class. Without shuffling, the training set could end up containing mostly the first few classes, which would badly skew learning.

</details>

---

## Step 2 — Visualise the Data (`show_data.py`)

Write a function `visualize_datalist` that accepts a list of rows (label + 784 pixel values) and displays a grid of images — one column per digit (0–9), multiple rows showing different instances of each digit.

<details>
<summary>Solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_datalist(datalist, instances_per_digit=3, digits=range(10), title=None):
    # Collect up to instances_per_digit images per digit
    instances = {digit: [] for digit in digits}
    for row in datalist:
        label = row[0]
        if label in instances and len(instances[label]) < instances_per_digit:
            instances[label].append(row[1:])
        if all(len(v) == instances_per_digit for v in instances.values()):
            break

    fig, axis = plt.subplots(
        instances_per_digit, len(digits),
        figsize=(2 * len(digits), 2 * instances_per_digit),
    )
    if title:
        fig.suptitle(title)

    for col, digit in enumerate(digits):
        for row in range(instances_per_digit):
            ax = axis[row, col] if instances_per_digit > 1 else axis[col]
            if row < len(instances[digit]):
                image = np.array(instances[digit][row]).reshape(28, 28)
                ax.imshow(image, cmap='gray')
                if row == 0:
                    ax.set_title(f"{digit}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()
```

Each image is stored as a flat vector of 784 values. `reshape(28, 28)` converts it back to the 2-D grid that `imshow` can display. `cmap='gray'` renders the image in greyscale — appropriate for single-channel pixel data.

The function collects instances by iterating through the datalist once, stopping early once it has enough samples for every digit, which keeps it efficient even for large datasets.

</details>

---

## Step 3 — Build the Forward Pass (`neural_net.py`)

Implement the sigmoid activation function and the `test` function that performs a forward pass through the network. The function takes an input vector and both weight matrices, and returns the output vector (10 values, one per digit class).

<details>
<summary>Solution</summary>

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def test(input_vector, w_input_hidden, w_hidden_output):
    # Reshape input to column vector for matrix multiplication
    x = input_vector.reshape(-1, 1)

    # Hidden layer
    hidden_inputs  = np.dot(w_input_hidden, x)
    hidden_outputs = sigmoid(hidden_inputs)

    # Output layer
    final_inputs  = np.dot(w_hidden_output, hidden_outputs)
    final_outputs = sigmoid(final_inputs)

    return final_outputs.flatten()
```

**Why reshape to a column vector?**
Matrix multiplication requires compatible shapes. With `w_input_hidden` of shape `(hidden_nodes, 784)`, the input must be `(784, 1)` for the dot product to produce `(hidden_nodes, 1)`. `.flatten()` at the end converts the output back to a 1-D array for convenience.

**Why sigmoid at every layer?**
Without a non-linear activation, stacking multiple linear layers collapses to a single linear transformation — no matter how many layers you add, the network could only learn linear boundaries. Sigmoid introduces the non-linearity needed to learn complex patterns.

**Reading the output:**
The output layer has 10 nodes — one per digit. The node with the highest activation is the predicted class: `np.argmax(output_vector)`.

</details>

---

## Step 4 — Train the Network (`train_net.py`)

Implement the `train` function that performs one forward pass, computes the error, and updates both weight matrices using backpropagation. The target vector encodes the correct label as 0.99 (not 1.0) and all other outputs as 0.01 (not 0.0).

<details>
<summary>Solution</summary>

```python
import numpy as np
from neural_net import sigmoid

def train(input_vector, label, w_input_hidden, w_hidden_output, learning_rate):
    x = input_vector.reshape(-1, 1)

    # --- Forward pass ---
    hidden_inputs  = np.dot(w_input_hidden, x)
    hidden_outputs = sigmoid(hidden_inputs)
    final_inputs   = np.dot(w_hidden_output, hidden_outputs)
    final_outputs  = sigmoid(final_inputs)

    # --- Target vector ---
    targets = np.zeros((10, 1)) + 0.01
    targets[label] = 0.99

    # --- Errors ---
    output_errors = targets - final_outputs
    hidden_errors = np.dot(w_hidden_output.T, output_errors)

    # --- Weight updates (backpropagation) ---
    # Hidden → output weights
    w_hidden_output += learning_rate * np.dot(
        (output_errors * final_outputs * (1.0 - final_outputs)),
        hidden_outputs.T,
    )
    # Input → hidden weights
    w_input_hidden += learning_rate * np.dot(
        (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        x.T,
    )

    return w_input_hidden, w_hidden_output
```

**Why 0.99 and 0.01 instead of 1 and 0?**
The sigmoid function never actually reaches 0 or 1 — it would require an infinitely large or small weight. Targeting 0.99 and 0.01 instead keeps the network in the sigmoid's active range and prevents weights from growing without bound.

**The weight update rule:**
Each weight is updated by:
```
ΔW = learning_rate · (error · σ'(output)) · input
```
Where `σ'(z) = σ(z) · (1 − σ(z))` is the sigmoid derivative. This is efficient because `σ(z)` was already computed in the forward pass — `final_outputs * (1.0 - final_outputs)` reuses it directly.

**Backpropagating the hidden error:**
The hidden layer has no direct target, so its error is estimated by multiplying the output errors back through the output weight matrix: `w_hidden_output.T · output_errors`. This distributes responsibility for the output error to each hidden node proportionally to its weight.

</details>

---

## Step 5 — Run the Full Pipeline (`run_net.py`)

Wire everything together: initialise weight matrices, train the network on all training samples for one epoch, run inference on the first test sample, print the output vector and predicted label, and display the corresponding image.

<details>
<summary>Solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from prepare_data import load_and_prepare_data
from neural_net import test
from train_net import train

np.random.seed(42)

# Network architecture
input_nodes  = 784   # 28×28 pixels
hidden_nodes = 100
output_nodes = 10    # digits 0–9
learning_rate = 0.3

# Initialise weights uniformly at random in [-0.5, 0.5]
w_input_hidden  = np.random.uniform(-0.5, 0.5, (hidden_nodes, input_nodes))
w_hidden_output = np.random.uniform(-0.5, 0.5, (output_nodes, hidden_nodes))

# Load data
training_data, test_data, training_labels, test_labels = load_and_prepare_data(
    "mnist_test.csv"
)

# Train for one epoch (one pass through all training samples)
for i in range(len(training_data)):
    input_vector = training_data[i]
    label        = int(training_labels[i])
    w_input_hidden, w_hidden_output = train(
        input_vector, label, w_input_hidden, w_hidden_output, learning_rate
    )

# Inference on the first test sample
first_test_input = test_data[0]
first_test_label = int(test_labels[0])
output_vector    = test(first_test_input, w_input_hidden, w_hidden_output)

print("Output vector for the first test element:")
print(output_vector)
print("Predicted label:", np.argmax(output_vector))
print("True label:     ", first_test_label)

# Display the image
plt.imshow(first_test_input.reshape(28, 28), cmap='gray')
plt.title(f"True label: {first_test_label}")
plt.axis('off')
plt.show()
```

**Weight initialisation:**
Weights are drawn from a uniform distribution in [−0.5, 0.5]. This breaks symmetry — if all weights started at zero, every hidden node would receive identical gradients and learn the same thing, making the hidden layer redundant.

**One epoch:**
The network sees each training sample exactly once. More epochs generally improve accuracy but risk overfitting. A single epoch is sufficient here to demonstrate the concept.

**Learning rate = 0.3:**
Controls the step size of each weight update. Too large and training becomes unstable (weights oscillate); too small and learning is very slow. 0.3 is a reasonable starting point for a sigmoid network on MNIST.

**`np.argmax(output_vector)`:**
The output layer produces 10 activation values. The index of the highest value is the predicted digit — the node the network is most confident about.

</details>
