# 📌 Social Media Comment Classification for Cyberbullying Detection

This project is a supervised ML pipeline to detect cyberbullying in social media comments.

---

## ✅ Project Structure

Social Media Comment Classification for Cyberbullying Detection
│
├── Data/
│ ├── cyberbullying_detection_dataset.csv
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ ├── y_test.csv
│
├── Training/
│ ├── training_notebook.ipynb
│ ├── preprocess_data.ipynb
│
├── Evaluation/
│ ├── evaluation_and_tuning.ipynb
│ ├── best_model_saving.ipynb
│
├── App/
│ ├── app.py
│ ├── model/
│ │ ├── model.pkl
│ │ ├── vectorizer.pkl
│ ├── templates/
│ │ ├── index.html
│ ├── static/
│ │ ├── css/
│ │ ├── js/
│
├── README.md
├── requirements.txt
├── python_version.txt
├── setup.exe (optional)

---

## ✅ Steps

### 1️⃣ Preprocessing  
- `preprocess_data.ipynb`  
- Cleans text, handles nulls, duplicates, outliers.
- TF-IDF comes later — text stays clean here.

### 2️⃣ Training  
- `training_notebook.ipynb` trains Logistic Regression, SVM, Random Forest, AdaBoost, Gradient Boosting.
- Checks baseline accuracy.

### 3️⃣ Evaluation & Tuning  
- `evaluation_and_tuning.ipynb` runs GridSearchCV for hyperparameter tuning.
- Shows confusion matrix, classification report, ROC AUC.

### 4️⃣ Best Model Saving  
- `best_model_saving.ipynb` picks best model by accuracy, saves `model.pkl` + `vectorizer.pkl`.

### 5️⃣ Flask App  
- `app.py` is the backend.  
- `index.html` is the UI.  
- Users enter comment + platform → prediction → result: **Bullying or Non-Bullying**.

---

## ✅ How to Run Flask App

```bash
cd App/
python app.py

Open http://127.0.0.1:5000