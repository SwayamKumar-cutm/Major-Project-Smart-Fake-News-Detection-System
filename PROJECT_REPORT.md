# Project Report: Smart Fake News Detection System

**Student:** Swayam Kumar  
**Roll No.:** AIML-A6/SEP-11833  
**Project Type:** Major Project

## Abstract
This project demonstrates text classification for news-like text using TF-IDF features and compares Logistic Regression with Multinomial Naive Bayes. A Streamlit interface provides an interactive demonstration.

## Objectives
- Clean and validate labeled text data.
- Convert text to TF-IDF features.
- Compare two supervised classification algorithms.
- Report accuracy, macro precision, macro recall, and macro F1.
- Persist a trained model and vectorizer with Joblib.
- Provide an interactive Streamlit application.

## Tools
Python, pandas, scikit-learn, Joblib, and Streamlit.

## Workflow
1. Validate the `text` and `label` fields.
2. Split the dataset into stratified training and test sets.
3. Fit TF-IDF on training text and transform the test text.
4. Train and compare Logistic Regression and Multinomial Naive Bayes.
5. Select the classifier with the strongest macro F1 on the supplied split.
6. Save model artifacts and metrics, then launch the Streamlit app.

## Dataset and responsible-use limitation
The included dataset is only an illustrative starter sample. It does not validate real-world fake-news detection. A larger, properly licensed and documented dataset is needed for meaningful evaluation. The model does not verify sources or establish factual truth; its predictions can be wrong.

## Conclusion
This project demonstrates an NLP classification pipeline and web interface. Any performance claims must be re-evaluated on an appropriate, representative dataset.