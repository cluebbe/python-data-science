# Import numpy for numerical operations
import numpy as np
# Import pandas for data manipulation
import pandas as pd
# Import train_test_split for splitting data
from sklearn.model_selection import train_test_split

def load_and_prepare_data(path):
    """
    Reads, splits, and normalizes the MNIST data.
    Args:
        path (str): Path to the CSV data file.
    Returns:
        tuple: (training_data, test_data, training_labels, test_labels)
    """
    # Read the CSV file into a pandas DataFrame without headers
    df = pd.read_csv(path, header=None)
    # Convert the DataFrame to a list of lists
    datalist = df.values.tolist()
    # Extract the first element of each row as labels
    label_list = [row[0] for row in datalist]
    # Extract the remaining elements of each row as data
    data_list = [row[1:] for row in datalist]
    # Split data into training and test sets (80/20 split)
    training_data, test_data, training_labels, test_labels = train_test_split(
        np.array(data_list),
        np.array(label_list),
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    # Normalize training data to [0.01, 1.0]
    training_data = np.asarray(training_data) / 255 * 0.99 + 0.01
    # Normalize test data to [0.01, 1.0]
    test_data = np.asarray(test_data) / 255 * 0.99 + 0.01
    # Return the split and normalized data and labels
    return training_data, test_data, training_labels, test_labels

if __name__ == "__main__":
    # Import the visualization function
    from show_data import visualize_datalist
    # Load and prepare data
    training_data, test_data, training_labels, test_labels = load_and_prepare_data(
        "mnist_test.csv"
    )
    # Combine labels and data for visualization (as in the original datalist format)
    training_datalist = [
        [int(label)] + list(vec) for label, vec in zip(training_labels, training_data)
    ]
    test_datalist = [
        [int(label)] + list(vec) for label, vec in zip(test_labels, test_data)
    ]
    # Display training data samples
    print("Displaying training data samples:")
    visualize_datalist(training_datalist, title="Training")
    # Display test data samples
    print("Displaying test data samples:")
    visualize_datalist(test_datalist, title="Test")