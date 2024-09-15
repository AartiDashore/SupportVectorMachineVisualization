# SupportVectorMachineVisualization

## Description
This project demonstrates how to implement a Support Vector Machine (SVM) classifier for binary classification and visualize the decision boundary, support vectors, and margins. The implementation uses the Iris dataset for training and classification. The visualization shows how SVM separates two classes by finding the optimal hyperplane and highlights the support vectors that define the decision boundary.

## Features
- SVM Classifier: Implement an SVM classifier using a linear kernel.
- Decision Boundary Visualization: Visualize the decision boundary, margins, and support vectors.
- Binary Classification: Train the model for two classes of the Iris dataset (Setosa and Versicolor).
- Customization: Easily modify the dataset or SVM parameters.

## Libraries
1. numpy: For numerical operations.
2. matplotlib: For data visualization and plotting the decision boundary.
3. scikit-learn: For loading the Iris dataset, SVM implementation, and splitting the dataset into training and testing sets.

## Dataset
The Iris dataset from scikit-learn is used. This dataset contains 150 samples of flowers, with 4 features (sepal length, sepal width, petal length, and petal width). However, for simplicity and visualization purposes:

- Only the first two features (sepal length and sepal width) are used.
- Only two classes (Setosa and Versicolor) are selected for binary classification.

## Explanation of Variables and Code:

### Data Loading and Preprocessing
We load the Iris dataset and filter the first two features and two classes (Setosa and Versicolor) for binary classification.

1. **`iris`**:
   - **Type**: Dataset
   - **Description**: The Iris dataset from `scikit-learn` is used in this example. We use only the first two features (sepal length and sepal width) for 2D visualization.
   - **Variable**: `iris = datasets.load_iris()`

2. **`X`**:
   - **Type**: `numpy` array
   - **Description**: Feature matrix, containing only the first two features (sepal length and sepal width) from the Iris dataset for easier 2D visualization. Here, we are also selecting only two classes (Setosa and Versicolor) for binary classification.
   - **Variable**: `X = iris.data[:, :2]`

3. **`y`**:
   - **Type**: `numpy` array
   - **Description**: Target labels for classification. We filter the target labels to include only two classes for binary classification (Setosa and Versicolor, where `y != 2`).
   - **Variable**: `y = iris.target`

### Split Data into Training and Testing Sets
We split the dataset into training and testing sets for model evaluation.

4. **`X_train`, `X_test`, `y_train`, `y_test`**:
   - **Type**: `numpy` arrays
   - **Description**: Split of the dataset into training and testing sets. We use `train_test_split` to divide the data, reserving 20% for testing and 80% for training.
   - **Variable**: `train_test_split(X, y, test_size=0.2, random_state=42)`

### Implement the SVM Classifier
An SVM classifier is initialized using a linear kernel and trained on the dataset.

5. **`svm`**:
   - **Type**: SVM Classifier Object
   - **Description**: The SVM model, initialized with a linear kernel. The `C` parameter is the regularization parameter, which controls the trade-off between maximizing the margin and minimizing the classification error.
   - **Variable**: `svm = SVC(kernel='linear', C=1.0)`

6. **`svm.fit(X_train, y_train)`**:
   - **Action**: Fits the SVM model to the training data. This step identifies the decision boundary, the support vectors, and the margins.

### Visualization of Decision Boundary
The function plot_svm_decision_boundary generates a plot showing the decision boundary, margins, and support vectors.

7. **`plot_svm_decision_boundary()`**:
   - **Description**: A function that visualizes the SVM decision boundary, margins, and support vectors.

   - **Key Variables Inside the Function**:
     - **`xx, yy`**: These variables represent a mesh grid used to create a continuous 2D surface over the feature space for plotting the decision boundary.
       - **Variable**: `xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))`
     - **`Z`**: This stores the decision function values, which are used to plot the decision boundary and margins. The decision boundary is the line where the decision function equals zero, while the margins are the lines where the decision function equals ±1.
       - **Variable**: `Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])`
     - **`model.support_vectors_`**: The support vectors identified by the SVM model. These points lie on or within the margin and are critical for defining the decision boundary.
       - **Variable**: `model.support_vectors_[:, 0]` and `model.support_vectors_[:, 1]`

8. **Visualization**:
   - **Decision Boundary**: The black line in the plot shows the decision boundary where the decision function equals zero.
   - **Margins**: The dashed blue lines represent the margins, where the decision function equals ±1. These margins are equidistant from the decision boundary.
   - **Support Vectors**: The points highlighted with circles are the support vectors. These points are crucial for defining the decision boundary.

### How to Run the Code:

1. **Install Required Libraries**:
   Install the necessary Python packages by running:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

2. **Run the Script**:
   Execute the Python script. It will train an SVM classifier on the Iris dataset and visualize the decision boundary, support vectors, and margins.

### Concepts:

1. **Support Vector Machines (SVM)**: 
   SVM is a supervised machine learning algorithm used for classification. It works by finding the hyperplane that best separates the data into different classes. The goal is to maximize the margin between the hyperplane and the nearest data points (support vectors).

2. **Support Vectors**: 
   These are the data points that lie closest to the decision boundary. These points define the margin and the optimal hyperplane.

3. **Margin**: 
   The distance between the decision boundary (hyperplane) and the support vectors. The SVM algorithm maximizes this margin.

4. **Decision Boundary**: 
   The line (in 2D) or hyperplane (in higher dimensions) that separates different classes. It is determined based on the support vectors.

5. **Kernel Function**: 
   In this example, a linear kernel is used, which is appropriate for linearly separable data. However, other kernels like polynomial or radial basis function (RBF) can be used for non-linearly separable data.


## Output:
![Output1.png](https://github.com/AartiDashore/SupportVectorMachineVisualization/blob/main/Output1.png)

## Conclusion:

This script provides a simple and clear implementation of an SVM classifier with visualization of the decision boundary, margins, and support vectors. It is a useful tool for understanding how SVM works in binary classification and how support vectors contribute to determining the decision boundary.
