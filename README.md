🧠 Sarcasm Detection using Machine Learning

This project focuses on detecting sarcasm in text using Natural Language Processing (NLP) and supervised machine learning algorithms — Naive Bayes, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).
The model analyzes textual data (news headlines and online comments) to determine whether a statement is sarcastic or not.


📌 Problem Statement

Sarcasm detection is a challenging task in Natural Language Processing because sarcastic sentences often convey the opposite of their literal meaning.
Traditional sentiment analysis systems struggle to interpret sarcasm accurately, leading to incorrect sentiment classification.
Hence, building a model that can effectively identify sarcasm is essential for improving NLP-based sentiment understanding.


🎯 Objective

To preprocess and clean text data from multiple sarcasm datasets.

To train machine learning models (Naive Bayes, SVM, KNN) for sarcasm detection.

To compare model accuracies and identify the best-performing algorithm.

To provide an interactive interface where users can input text and get sarcasm predictions.


📂 Dataset Information

This project uses publicly available datasets for sarcasm detection:

🔗 News Headlines Dataset for Sarcasm Detection

Additionally, the Balanced Reddit Sarcasm Dataset (train-balanced-sarcasm.csv) can be included for better performance.

⚠️ Note: The datasets are not uploaded to GitHub due to large file size limits.
Instead, upload dataset from (/content/train-balanced-sarcasm.csv) in kaggle.



🧹 Data Preprocessing Steps

Convert text to lowercase

Remove URLs, mentions, hashtags, and special characters

Remove English stopwords

Convert cleaned text into TF-IDF features for training


🤖 Algorithms Used
Algorithm  - Description	Accuracy
Naive Bayes	Fast probabilistic model suitable for text classification  -	~82%
SVM (LinearSVC)	Strong linear classifier for high-dimensional text data  -	~86%
K-Nearest Neighbors	Instance-based model for comparison-based classification  -	~80%


📊 Model Comparison

A bar graph is generated to compare the performance of all three models visually.

plt.figure(figsize=(7,4))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Accuracy Comparison (All 3 Datasets Combined)")
plt.ylabel("Accuracy")
plt.show()


💬 Interactive Prediction (Gradio App)

A Gradio interface allows users to test the model interactively:

def predict_sarcasm(text):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = best_model.predict(features)[0]
    return "😏 Sarcastic" if pred == 1 else "🙂 Not Sarcastic"

import gradio as gr
interface = gr.Interface(fn=predict_sarcasm, inputs="text", outputs="text", title="Sarcasm Detection")
interface.launch(share=True)


🧩** Development of Dataset (Workflow Diagram)**

Below is the flow of dataset development and usage:

Raw Kaggle + Reddit Datasets
        │
        ▼
  Text Cleaning & Preprocessing
        │
        ▼
   TF-IDF Feature Extraction
        │
        ▼
 Train-Test Split (80:20)
        │
        ▼
 Model Training (SVM, NB, KNN)
        │
        ▼
 Evaluation & Prediction


🚀 **Results**

Best Model: SVM (LinearSVC)

Accuracy Achieved: ~86%

Performs consistently well across multiple sarcasm datasets.


👩‍💻** How to Run in Google Colab**

Open the .ipynb file in Google Colab.

Run all cells in sequence.

Upload dataset from Kaggle.

Wait for model training and evaluation to complete.

Use the Gradio app to test custom text inputs.


📜** License**

This project is open-source and available for educational and research purposes.

Would you like me to add a short project description and GitHub tags section (for your repo top part)? It makes your project look professional on GitHub.
