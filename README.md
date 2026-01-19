# AI-ML-Internship
Repository for AI & ML Internship Tasks.

This repository contains the completed tasks assigned as part of an AI & Machine Learning Internship.
The work focuses on building practical, end-to-end machine learning projects, covering both numerical data classification and text-based spam detection using NLP.

The goal of this repository is to demonstrate:

Clear understanding of ML workflows

Proper use of datasets and evaluation metrics

Clean, readable, and reproducible code

Well-documented results and explanations

📌 Tasks Overview
✅ Task 1: Iris Flower Classification

A supervised machine learning model is built to classify iris flowers into three species:

Setosa

Versicolor

Virginica

The classification is based on four numerical features:

Sepal length

Sepal width

Petal length

Petal width

A Logistic Regression classifier is used for this task.

✅ Task 2: SMS Spam Detection

A text classification system is developed to identify whether an SMS message is:

Spam

Ham (Not Spam)

This task uses Natural Language Processing (NLP) techniques along with a Naive Bayes classifier to detect spam messages effectively.

📂 Repository Structure
AI-ML-Internship/
│
├── Task-1_Iris_Classification/
│   ├── iris_model.py
│   └── Iris_Report.docx
│
├── Task-2_Spam_Detection/
│   ├── spam_detector.py
│   ├── SMSSpamCollection
│   └── Spam_Report.docx
│
├── Combined_Report.docx
└── requirements.txt

🧠 Task 1 Details – Iris Classification
Dataset

Iris Dataset (available via scikit-learn)

150 samples

4 numerical features

3 target classes

Methodology

Dataset loading using scikit-learn

Train-test split (80:20)

Logistic Regression model training

Model evaluation using:

Accuracy score

Confusion matrix

Result

Accuracy achieved: 100%

No misclassification observed on the test set

This task demonstrates a clean and effective implementation of a supervised learning pipeline.

🧠 Task 2 Details – SMS Spam Detection
Dataset

SMS Spam Collection Dataset (UCI Machine Learning Repository)

Contains labeled SMS messages (spam / ham)

Text-based dataset

Methodology

Dataset loading (tab-separated format)

Text vectorization using TF-IDF

Train-test split (80:20)

Classification using Multinomial Naive Bayes

Model evaluation using:

Accuracy

Confusion matrix

Precision, Recall, and F1-score

Result

Accuracy achieved: ~98%

Zero false positives (no genuine message marked as spam)

Strong balance between precision and recall

This task demonstrates practical application of NLP techniques in machine learning.

📊 Model Performance Summary
Task	Model Used	Accuracy
Iris Classification	Logistic Regression	100%
SMS Spam Detection	Naive Bayes (TF-IDF)	~98%
🛠️ Technologies & Libraries Used

Python

NumPy

Pandas

Scikit-learn

Natural Language Processing (TF-IDF)

All required dependencies are listed in the requirements.txt file.

▶️ How to Run the Code
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run Iris Classification
python Task-1_Iris_Classification/iris_model.py

3️⃣ Run Spam Detection
python Task-2_Spam_Detection/spam_detector.py

📄 Documentation

Each task has its own detailed report explaining:

Objective

Dataset

Methodology

Results

A Combined_Report.docx is also included for overall submission.
