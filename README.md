# 🩺 Breast Cancer Detection using Machine Learning

This project uses **Machine Learning** to predict whether a tumor is **benign** or **malignant** based on the Breast Cancer Wisconsin dataset.  
It demonstrates a complete ML pipeline — from **data preprocessing** to **model training**, **evaluation**, and **deployment** using Flask / Streamlit.

---

## 📊 Project Overview
Breast cancer is one of the most common cancers worldwide.  
Early detection through ML models can help improve diagnosis and save lives.

This project includes:
- Data loading and preprocessing  
- Feature scaling and cleaning  
- Model training (Logistic Regression / Random Forest / SVM)  
- Evaluation metrics  
- Web API or UI for predictions

---

## 🧠 Technologies Used
- Python 3.10+  
- pandas, numpy, scikit-learn  
- matplotlib, seaborn  
- Flask or Streamlit  
- Jupyter Notebook / VS Code

---

## 🧩 Folder Structure
```
breast-cancer-wisconsin/
│
├── data_pipeline/
│   ├── load_data.py          # Loads and cleans dataset
│   ├── preprocess.py         # Scales and prepares data
│
├── models/                   # Trained models (.pkl or .h5)
├── api/ or app/              # Flask app or API
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── .gitignore                # Ignored files (like venv/)
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```
git clone https://github.com/<your-username>/breast-cancer-detection.git
cd breast-cancer-detection
```

### 2️⃣ Create a Virtual Environment
```
python -m venv venv
```

### 3️⃣ Activate the Environment
```
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 4️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 5️⃣ Run the Project
If Flask app:
```
python app.py
```

If Streamlit app:
```
streamlit run app.py
```

---

## 📈 Model Performance
| Metric | Score |
|--------|--------|
| Accuracy | 97% |
| Precision | 96% |
| Recall | 95% |

*(Replace these with your actual results.)*

---

## 🧬 Dataset
**Source:** [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
Features are computed from digitized images of fine needle aspirates (FNA) of breast masses.

---

## 🚀 Future Improvements
- Add Deep Learning (TensorFlow / PyTorch) models  
- Deploy to Render / Hugging Face / Streamlit Cloud  
- Add real-time prediction API  

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!  
Feel free to fork this repo and submit a pull request.

---

## 🧑‍💻 Author
**Divyansh Aggarwal**  
📍 SRM Institute of Science and Technology  

---

## 📜 License
This project is licensed under the **MIT License** —
