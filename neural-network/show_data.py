# Import pandas for data manipulation
import pandas as pd
# Import numpy for numerical operations
import numpy as np
# Import matplotlib for plotting images
import matplotlib.pyplot as plt

def visualize_datalist(
    datalist, instances_per_digit=3, digits=range(10), title=None
):
    """
    Visualize instances_per_digit images for each digit in datalist.
    Args:
        datalist (list): List of data rows, each row with label as first element and 784 pixel values.
        instances_per_digit (int): Number of images to show per digit.
        digits (iterable): Digits to visualize (default: 0-9).
        title (str or None): Title for the figure (default: None).
    """
    # Create a dictionary to collect up to instances_per_digit for each digit
    instances = {digit: [] for digit in digits}
    # Iterate over all rows in the datalist
    for row in datalist:
        # The first element is the label (digit)
        label = row[0]
        # Add the image data if we need more instances for this digit
        if label in instances and len(instances[label]) < instances_per_digit:
            instances[label].append(row[1:])
        # Stop if we have enough instances for all digits
        if all(len(v) == instances_per_digit for v in instances.values()):
            break
    # Create the figure and axes
    fig, axis = plt.subplots(
        instances_per_digit, len(digits),
        figsize=(2 * len(digits), 2 * instances_per_digit)
    )
    # Set the figure title if provided
    if title:
        fig.suptitle(title)
    # For each digit (column)
    for col, digit in enumerate(digits):
        # For each instance (row)
        for row in range(instances_per_digit):
            # Select the correct subplot
            ax = axis[row, col] if instances_per_digit > 1 else axis[col]
            if row < len(instances[digit]):
                # Convert the image data to a 28x28 array
                image = np.array(instances[digit][row]).reshape(28, 28)
                # Show the image in grayscale
                ax.imshow(image, cmap='gray')
                # Set the column title to the digit
                if row == 0:
                    ax.set_title(f"{digit}")
            else:
                # Hide the axis if no image
                ax.axis('off')
            # Hide axis ticks
            ax.axis('off')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Display the plot
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Path to the MNIST CSV data file
    path = r"mnist_test.csv"
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(path, header=None)
    # Convert the DataFrame to a list of lists
    datalist = df.values.tolist()
    # Visualize the datalist
    visualize_datalist(datalist)
