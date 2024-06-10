# lab2
# Breast Cancer Wisconsin Diagnostic Dataset Analysis

This repository contains an analysis of the Breast Cancer Wisconsin Diagnostic dataset using two machine learning models: RandomForestClassifier and LogisticRegression. The dataset is used to classify tumors as either malignant (cancerous) or benign (non-cancerous).

## Introduction
The aim of this project is to evaluate the performance of two machine learning models in classifying breast cancer tumors. The performance metrics include accuracy, precision, recall, F1-score, and confusion matrices.

## Dataset
The Breast Cancer Wisconsin Diagnostic dataset is publicly available and consists of features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe the characteristics of the cell nuclei present in the image.

## Models
Two models were used in this analysis:
1. **RandomForestClassifier**
2. **LogisticRegression** (inside model2 branch of the github repository)

## Results
### RandomForestClassifier
- **Accuracy**: 0.96

#### Confusion Matrix
```
[[70  1]
 [ 3 40]]
```

#### Classification Report
```
               precision    recall  f1-score   support

           0       0.96      0.99      0.97        71
           1       0.98      0.93      0.95        43

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114
```

### LogisticRegression
- **Accuracy**: 0.96

#### Confusion Matrix
```
[[70  1]
 [ 4 39]]
```

#### Classification Report
```
               precision    recall  f1-score   support

           0       0.95      0.99      0.97        71
           1       0.97      0.91      0.94        43

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114
```

## Conclusion
Both RandomForestClassifier and LogisticRegression models have demonstrated excellent performance on the Breast Cancer Wisconsin Diagnostic dataset, achieving an identical accuracy of 0.96. However, RandomForestClassifier slightly outperforms LogisticRegression in terms of precision, recall, and F1-score for the malignant class.

### Key Insights:
- RandomForestClassifier misclassified 4 samples (3 false negatives, 1 false positive).
- LogisticRegression misclassified 5 samples (4 false negatives, 1 false positive).

Both models are highly effective for predicting breast cancer and can be confidently used in clinical settings to aid in diagnosis.

## How to Use
1. **Clone the repository**:
   ```
   git clone https://github.com/Rajinderk1/lab2/
   ```
2. **Navigate to the project directory**:
   ```
   cd lab2
   ```
3. **Run the ipynb file on Jupyter Notebook**
   ```

## References
- [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/)
