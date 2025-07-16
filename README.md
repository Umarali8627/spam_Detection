# Email Spam Detection

This project implements a machine learning model to classify emails as spam or ham (non-spam) using the SMS Spam Collection dataset. It employs Multinomial Naive Bayes with TF-IDF vectorization for text preprocessing, achieving high accuracy in spam detection. The project is built in Jupyter Notebooks, includes data visualizations, and supports single email predictions, making it suitable for practical spam filtering applications.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Example Output](#example-output)
- [Visualizations](#visualizations)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [design](#Design)

## Project Overview
This project develops a spam detection system using:
- **Dataset**: SMS Spam Collection (`spam.csv`) with 5,572 labeled messages (4,825 ham, 747 spam).
- **Model**: Multinomial Naive Bayes, trained on TF-IDF features for efficient text classification.
- **Preprocessing**: TF-IDF vectorization to convert text messages into numerical features.
- **Functionality**:
  - Train and evaluate the model in `Email_Spam_Detection.ipynb`.
  - Predict spam/ham for new emails in `Predict_Email_Spam.ipynb`.
  - Visualize data distributions and model performance (e.g., confusion matrix).
- **Goal**: Provide a reusable, well-documented solution for spam detection, hosted on GitHub for collaboration and sharing.

## Dataset
- **Source**: Kaggle
- **File**: `spam.csv`
- **Columns**:
  - `Category`: Label (`ham` or `spam`).
  - `Message`: Raw text of the email/SMS.
- **Preprocessing**:
  - Loaded with `pandas`.
  - No missing values.
  - Text transformed using `TfidfVectorizer` from `scikit-learn`.
  - 80/20 train-test split for model evaluation.

## Features
- Accurate spam detection using Multinomial Naive Bayes.
- Visualizations of class distribution and model performance (confusion matrix).
- Saved model and vectorizer for reusable predictions.
- Single email prediction functionality with spam probability output.
- Modular code structure in Jupyter Notebooks.
- GitHub repository with clear documentation for easy setup.

## Requirements
- Python 3.8+
- Libraries:
  - `pandas`
  - `scikit-learn`
  - `matplotlib
  - `seaborn`
  - `wordcloud`
    ## Design
    -for before runining the index.html file run the app.py to get the api_url to get the model predication
    --app .py is used
    -flask
    -pickle
    index.html
    -pure html
  
- Jupyter Notebook or JupyterLab
- Git (for cloning the repository)

