📊 Sentiment Analysis on Flipkart Reviews

This project focuses on building a Machine Learning–based Sentiment Analysis system to classify Flipkart product reviews as Positive or Negative using Natural Language Processing (NLP) techniques.

🚀 Project Overview

Customer reviews play a crucial role in understanding product quality and user satisfaction.
In this project, I developed an end-to-end sentiment classification pipeline that analyzes real Flipkart reviews and predicts sentiment using a trained ML model.

🧠 Key Features
Text preprocessing and cleaning
Feature extraction using TF-IDF Vectorizer
Machine Learning–based sentiment classification
Saved and reusable trained model
Predict sentiment for new/unseen reviews

🛠 Tech Stack & Tools
Programming Language: Python

Libraries:
Pandas
NumPy
Scikit-learn
NLTK

NLP Techniques:
Text Cleaning
Tokenization

📂 Project Structure
Sentiment-Analysis-Flipkart/
│
├── data.csv                    # Dataset containing Flipkart reviews
├── sentimental_analysis.ipynb  # Model training & analysis notebook
├── sentiment_app.py            # Script for sentiment prediction
├── sentiment_model.pkl         # Trained ML model
├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt            # Required dependencies
└── README.md                   # Project documentation

⚙️ Installation & Setup
Clone the repository
git clone https://github.com/Nani769123/Sentiment-Analysis-Flipkart.git
Navigate to the project directory
cd Sentiment-Analysis-Flipkart
Install required dependencies
pip install -r requirements.txt

▶️ How to Run the Project
Run the Jupyter Notebook to explore data and model training:
jupyter notebook sentimental_analysis.ipynb

Run the sentiment prediction script:
python sentiment_app.py

📌 Sample Output
Input:
“The product quality is amazing and worth the price.”

Output:
✅ Positive Review

📈 What I Learned
Practical implementation of NLP concepts
Feature extraction using TF-IDF
Building and saving ML models
End-to-end machine learning workflow
Handling real-world text data

🔮 Future Improvements
Add neutral sentiment classification
Improve accuracy using deep learning models (LSTM / BERT)
Deploy as a web app using Streamlit or Flask
Add visualization dashboards

🔗 GitHub Repository
👉 https://github.com/Nani769123/Sentiment-Analysis-Flipkart

🤝 Connect With Me
If you have feedback, suggestions, or collaboration ideas, feel free to connect!

⭐ If you found this project helpful, don’t forget to star the repo!
TF-IDF Vectorization
Model Storage: Pickle (.pkl)
