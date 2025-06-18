
# Final Project 4.3

## 🚀 Overview
This repository contains code and notebooks for a two-part machine-learning pipeline:
1. **No-Show Prediction** – a classification model to predict which patients will miss their appointments.  
2. **Patient Sentiment Analysis** – a Transformer-based model to gauge patient sentiment from text.

## 📁 Repository Structure
Final-Project-4-3/
│
├── Data/                                  # raw & processed datasets
│   └── [dataset.csv & dataset1.csv]
│
├── EDA.ipynb                              # exploratory data analysis notebook
├── Supervised learning.ipynb              # model training & evaluation notebook
│
├── transformers/                          # custom transformer code 
│
├── models/                                # saved model checkpoints
│   └── transformers_patient_sentiment/    # trained sentiment-analysis models
│
├── app.py                                 # 
├── requirements.txt                       # Python dependencies
└── README.md                              # this file
## ⚙️ Prerequisites & Setup
1. **Python version**: ≥ 3.8  
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

## Exploratory Data Analysis

## Model findings
    
- **XGBoost** achieved the highest accuracy (0.85) and ROC-AUC (0.82).  
- **Random Forest** came in second (0.84 acc, 0.79 AUC).  
- **Logistic Regression**: 0.81 accuracy, 0.74 AUC. 


## License

This project is released under the MIT License.



MIT License

Copyright (c) [2025] [Kevin Hooman,Swapnil Patil, Dillard Holder]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.