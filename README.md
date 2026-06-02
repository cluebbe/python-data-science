# Python Data Science Tutorials

A collection of hands-on workshop tutorials covering core machine learning and data science algorithms using scikit-learn and Python. Each tutorial consists of a `.md` file with step-by-step tasks and collapsible solutions, and a `.py` file with the complete working code.

---

## Index

### Supervised Learning

| Tutorial | Description | Files |
|---|---|---|
| [Decision Trees](DECISION_TREE_TUTORIAL.md) | Build and visualise a decision tree classifier on the Iris dataset. Covers tree structure, feature importance, overfitting via depth control, and single predictions. | [.md](DECISION_TREE_TUTORIAL.md) · [.py](decision_tree_tutorial.py) |
| [Logistic Regression](LOGISTIC_REGRESSION_TUTORIAL.md) | Train a logistic regression classifier on the breast cancer dataset. Covers feature scaling, evaluation metrics (accuracy, ROC-AUC, confusion matrix), coefficients, regularisation (C), partial dependence plots, and Shapley value waterfall charts. | [.md](LOGISTIC_REGRESSION_TUTORIAL.md) · [.py](logistic_regression_tutorial.py) |
| [Random Forest](RANDOM_FOREST_TUTORIAL.md) | Extend decision trees to a random forest ensemble. Covers bagging, feature importance, out-of-bag error, and comparison with a single decision tree. | [.md](RANDOM_FOREST_TUTORIAL.md) · [.py](random_forest_tutorial.py) |
| [Neural Network](neural-network/NEURAL_NETWORK_TUTORIAL.md) | Build and train a neural network from scratch on the MNIST handwritten digit dataset. Covers forward propagation, backpropagation, activation functions, and digit recognition. | [.md](neural-network/NEURAL_NETWORK_TUTORIAL.md) · [.py](neural-network/neural_net.py) |

### Unsupervised Learning

| Tutorial | Description | Files |
|---|---|---|
| [K-Means Clustering](KMEANS_TUTORIAL.md) | Cluster synthetic data using K-means. Covers feature scaling, choosing K with the elbow and silhouette methods, visualising clusters, and the effect of different K values. | [.md](KMEANS_TUTORIAL.md) · [.py](kmeans_tutorial.py) |
| [Agglomerative Clustering](AGGLOMERATIVE_TUTORIAL.md) | Cluster data using hierarchical agglomerative clustering. Covers the dendrogram, Ward linkage, the effect of different linkage criteria on non-convex data (`make_moons`), and assigning new samples. | [.md](AGGLOMERATIVE_TUTORIAL.md) · [.py](agglomerative_tutorial.py) |

---

## Prerequisites

- Python 3.9 or newer
- pip

Install all dependencies:

```bash
pip install numpy matplotlib scikit-learn scipy
```

Each tutorial's `.md` file includes its own environment setup section with step-by-step instructions.

---

## How to Use

Each tutorial is self-contained and can be worked through independently. The recommended order follows the index above — supervised learning builds progressively, and unsupervised learning can be read in any order.

1. Open the `.md` file and read the background section
2. Work through each step — attempt the task before expanding the solution
3. Run the `.py` file to see all steps executed end to end
