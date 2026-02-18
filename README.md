# Fake News Detection using Machine Learning

## Project Overview

The rapid spread of misinformation through digital platforms has become a major concern. This project focuses on building a machine learning model that can classify news articles as Real or Fake based on their textual content.

The goal of this project was to understand how Natural Language Processing techniques can be applied to solve real-world classification problems and to build a reliable predictive model.

---

## Objective

- To preprocess and clean textual news data.
- To convert text into numerical features using vectorization techniques.
- To train and evaluate classification models.
- To select the best-performing model based on evaluation metrics.

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Natural Language Processing (NLP)  
- TF-IDF Vectorization  
- Logistic Regression / Passive Aggressive Classifier  

---

## Methodology

1. Data Cleaning  
   - Removed punctuation, special characters, and unnecessary whitespace.  
   - Converted text to lowercase.  
   - Removed stopwords where necessary.  

2. Feature Extraction  
   - Applied TF-IDF Vectorization to transform text into numerical features.

3. Model Training  
   - Split the dataset into training and testing sets.
   - Trained classification models on the processed data.

4. Model Evaluation  
   - Evaluated using accuracy, precision, recall, and confusion matrix.
   - Compared models and selected the best-performing one.

---

## Model Performance

- Accuracy: 94%  
- Precision: 93%  
- Recall: 95%  

The selected model demonstrated strong generalization performance on unseen test data.

---

## Dataset

The dataset consists of labeled news articles categorized as Real or Fake.

Due to GitHub file size limitations, the complete dataset is not included in this repository. However, the entire preprocessing pipeline and model training workflow are implemented in the code.

---

## Key Learnings

- Practical experience with text preprocessing in NLP.
- Understanding feature extraction using TF-IDF.
- Building and evaluating classification models.
- Interpreting evaluation metrics for performance improvement.
- Handling real-world unstructured text data.

---

## Future Enhancements

- Deploy the model as a web application using Flask.
- Experiment with deep learning approaches such as LSTM or transformer-based models.
- Integrate real-time news input functionality.

---

