# 🔐 DeepNIDS – Deep Learning Based Network Intrusion Detection System

DeepNIDS is a **Deep Learning–based Network Intrusion Detection System** developed using the **NSL-KDD dataset**.
The system detects **normal and malicious network traffic** using a **Deep Neural Network (DNN)** and provides a **web-based user interface using Flask** for real-time analysis, visualization, and reporting.

---

## 📌 Project Objectives

* Detect network intrusions using **deep learning techniques**
* Handle **imbalanced attack data** effectively
* Provide a **user-friendly web interface**
* Visualize results using **graphs**
* Generate **downloadable PDF reports**
* Demonstrate a **complete end-to-end IDS pipeline**

---

## 🧠 System Overview

The system consists of four main modules:

1. **Data Preprocessing**
2. **Deep Learning Model**
3. **Evaluation & Visualization**
4. **Flask Web Application**

---

## 📂 Project Structure

```
NIDSProject---new/
│
├── main.py                    # Model training & evaluation
├── app.py                     # Flask web application
├── README.md
│
├── NSL_KDD.csv                # Dataset
│
├── preprocessing/
│   ├── data_cleaning.py
│   └── feature_engineering.py
│
├── models/
│   └── dnn_model.py
│
├── training/
│   └── callbacks.py
│
├── evaluation/
│   └── evaluate_model.py
│
├── saved_models/
│   └── dnn_final_model.h5
│
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── index.html
│   ├── result.html
│   └── admin.html
│
└── static/
    ├── style.css
    └── dark.css
```

---

## 📊 Dataset Used

* **Dataset Name:** NSL-KDD
* **Records:** 125,972
* **Features:** 41
* **Classes:**

  * Normal (0)
  * Attack (1)

The dataset is widely used for evaluating intrusion detection systems and addresses issues present in the original KDD’99 dataset.

---

## ⚙️ Data Preprocessing

* Removed duplicates and checked for missing values
* Encoded categorical features (`protocol`, `service`, `flag`)
* Converted multi-class labels into **binary classification**
* Applied **StandardScaler** for feature normalization
* Computed **class weights** to handle class imbalance

---

## 🤖 Deep Learning Model

### Model Type:

**Deep Neural Network (DNN)**

### Architecture:

* Input Layer: 41 features
* Hidden Layers:

  * Dense (128) + ReLU
  * Dense (64) + ReLU
  * Dense (32) + ReLU
* Dropout layers to prevent overfitting
* Output Layer: 1 neuron with **Sigmoid activation**

### Training Details:

* Optimizer: Adam
* Loss Function: Binary Cross-Entropy
* Batch Size: 128
* Epochs: 50
* Early Stopping enabled

---

## 📈 Model Evaluation

The trained model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The model achieves **high accuracy and recall**, indicating effective intrusion detection with minimal false positives.

---

## 🌐 Flask Web Application

The Flask UI allows users to:

* Login as Admin
* Upload network traffic CSV files
* Perform intrusion detection
* View results visually
* Download detection reports as PDF

### Key Features:

* 🔐 Login & Admin Dashboard
* 📊 Pie Chart & Bar Chart (Attack vs Normal)
* 🌙 Dark Mode
* 📄 PDF Report Generation
* 🎨 Bootstrap-based responsive UI

---

## 📊 Visualization

* **Pie Chart:** Traffic distribution (Normal vs Attack)
* **Bar Chart:** Comparison of attack and normal records
* Charts are rendered using **Chart.js**

---

## 📄 PDF Report

The system generates a **downloadable PDF report** containing:

* Total records
* Normal traffic count
* Attack traffic count
* Model and dataset details

This is useful for documentation and security audits.

---

## 🛠 Technologies Used

* **Python 3.10**
* **TensorFlow / Keras**
* **Scikit-learn**
* **Pandas & NumPy**
* **Flask**
* **Chart.js**
* **Bootstrap**
* **ReportLab (PDF generation)**

---

## ▶️ How to Run the Project

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install tensorflow==2.13.0 flask pandas numpy scikit-learn reportlab
```

### 3️⃣ Train the Model

```bash
python main.py
```

### 4️⃣ Run Flask App

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## 🎓 Academic Relevance

This project demonstrates:

* Practical application of deep learning
* Cybersecurity implementation
* Full-stack integration (ML + Web)
* Industry-relevant IDS design

It is suitable for:

* Diploma / B.Tech Final Year Projects
* Cybersecurity Demonstrations
* Machine Learning Case Studies

---

## 🧠 Viva-Ready Summary

> “DeepNIDS uses a deep neural network trained on the NSL-KDD dataset to accurately classify network traffic as normal or malicious. A Flask-based interface enables real-time detection, visualization, and report generation, making the system practical and user-friendly.”

---

## 📌 Future Enhancements

* Real-time packet capture
* Multi-class attack classification
* Cloud deployment
* REST API integration
* User-specific detection history

---

## 👨‍💻 Author

**Team of DeepNIDS**
Diploma Project – Network Intrusion Detection System
