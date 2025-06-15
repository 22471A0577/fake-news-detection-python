# Fake News Detection using Python

This is a simple machine learning project I created to detect whether a news article is fake or real. I used Python and a dataset from Kaggle that contains both fake and real news articles.

## Dataset

The dataset contains two CSV files:
- Fake.csv – Contains fake news articles
- True.csv – Contains real news articles

I combined them into one dataset and labeled them:
- 0 = Fake
- 1 = Real

## Technologies and Libraries Used

- Python
- Pandas
- NumPy
- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

## Steps in the Project

1. Loaded the fake and real news data
2. Added labels and combined the datasets
3. Converted the text to numeric form using TF-IDF
4. Split the data into training and test sets
5. Trained a Logistic Regression model
6. Checked the model's accuracy
7. Tested the model with custom news inputs

## How to Run

1. Install the required libraries:pip install pandas numpy scikit-learn
2. run the python script
3. the output will show the model accuracy and prediction result for a test input
4. Model Accuracy : 0.98 prediction(1= Real, ) = Fake):0

