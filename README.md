# Breast Cancer Classification

This project demonstrates the classification of breast cancer using machine learning algorithms. I have used four ML algorithms - K-Nearest Neighbors, Support Vector Machines, Decision Trees, and Random Forests. The accuracy scores and confusion matrices are presented for each algorithm.

## Data Description

The data set contains 569 samples of malignant and benign breast tumor cells. The data is in CSV format and contains 30 attributes, including the ID of the patient, the diagnosis, and the patient's age.
#this is where I downloaded the dataset 
#https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/download?datasetVersionNumber=2

## Code Execution

The code can be executed using any Python IDE. It involves importing the necessary libraries, reading the data from the CSV file, preprocessing the data, splitting the data into training and testing sets, and finally, applying the ML algorithms.

The results are presented in the form of accuracy scores, confusion matrices, and heatmaps for each algorithm. The comparison of the algorithms' performance is done using a bar plot.

### Running the code

To run the code, you can follow these steps:

1. Import the necessary libraries, including pandas, matplotlib, seaborn, sklearn.
2. Load the breast cancer dataset from the CSV file using pandas.
3. Preprocess the data by encoding categorical features using LabelEncoder.
4. Split the data into training and testing sets using train_test_split from sklearn.model_selection.
5. Apply the ML algorithms by fitting the training data and predicting the test data.
6. Calculate the accuracy scores and confusion matrices for each algorithm.
7. Plot the results using heatmaps and bar plots.

## Requirements

The following Python libraries are required to run the code:

- pandas
- matplotlib
- seaborn
- sklearn

## References
1. https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/download?datasetVersionNumber=2
2. scikit-learn User Guide: Support Vector Machines (https://scikit-learn.org/stable/modules/svm.html)
3. scikit-learn User Guide: K-Nearest Neighbors (https://scikit-learn.org/stable/modules/neighbors.html)
4. scikit-learn User Guide: Decision Trees (https://scikit-learn.org/stable/modules/tree.html)
5. scikit-learn User Guide: Random Forests (https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)

Please note that this README is not exhaustive and should be considered a starting point for understanding the project. For any further questions or concerns, feel free to contact me.
Kiplimo Victor
