# AI-ML-Internship
Repository For AI & ML Internship Tasks.

README — Task 2: SMS Spam Detection
📩 SMS Spam Detection (Natural Language Processing)

This project is part of the AI & Machine Learning Internship tasks.
The objective of Task 2 is to build a text classification system that can accurately identify spam messages.

📌 Problem Statement

Given an SMS message, classify it as:
Ham (legitimate message)
Spam (unwanted / promotional message)

🧠 Approach & Methodology

Used the SMS Spam Collection dataset
Performed text preprocessing and vectorization
Applied Logistic Regression for binary classification
Evaluated performance using standard NLP metrics

🛠️ Technologies Used

Python
scikit-learn
Natural Language Processing (NLP) techniques

📊 Model Evaluation

The model was evaluated using:
Accuracy
Confusion Matrix
Precision, Recall, and F1-score

Key Results:

Accuracy: ~98%
Zero false positives for legitimate (ham) messages
Strong balance between precision and recall for spam detection

📂 Files in This Branch
Task-2_Spam_detector/
├── spam_detector.py     # Spam classification code
├── SMSSpamCollection    # Dataset file
├── report.docx          # Detailed task documentation
├── README.md            # Project description

▶️ How to Run:
run the commands in the powershell (assuming pip is already install in the system):

pip install scikit-learn numpy
python spam_detector.py

✅ Conclusion

This task demonstrates:

Practical application of NLP concepts.
Effective text classification using ML.
Proper evaluation of real-world datasets.
