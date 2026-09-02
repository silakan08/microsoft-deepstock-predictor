# DeepStock Predictor: PyTorch Stock Price Forecasting Engine (LSTM vs. GRU)

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python)
![Live Dashboard](https://img.shields.io/badge/Live%20Web%20Dashboard-Active-0078D4?style=flat&logo=vercel)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

🌐 **Live Interactive Web Dashboard:** [https://microsoft-deepstock-predictor-1lr4.vercel.app/](https://microsoft-deepstock-predictor-1lr4.vercel.app/)

---

## 📌 Executive Overview
This repository contains a production-grade Time-Series Forecasting and Deep Learning evaluation engine built for the Microsoft Summer Voluntary Internship program. 

The project implements, trains, and evaluates two recurrent neural network architectures—**Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)**—using **PyTorch** to predict Amazon (`AMZN`) daily stock closing prices over a 12-year period (2012–2024).

In addition to the core PyTorch ML engine and runnable Jupyter Notebook, this project includes an **[Interactive Web Dashboard](https://microsoft-deepstock-predictor-1lr4.vercel.app/)** for live metric comparisons and an **Ethical AI & Limitations Analysis** based on Princeton University's *"AI Snake Oil"* framework.

---

## 🏗️ Repository Architecture

```
microsoft_stock_prediction/
├── data/
│   └── AMZN_stock_data.csv              # Official historical stock prices (2012-2024)
├── src/
│   ├── __init__.py
│   ├── fetch_data.py                    # Downloads official Yahoo Finance stock data
│   ├── data_loader.py                   # Scaling, sliding window, Train/Test split, PyTorch Tensors
│   ├── models.py                        # PyTorch LSTMModel & GRUModel architectures (nn.Module)
│   ├── train.py                         # Training loops, loss tracking, time profiling
│   └── evaluate.py                      # MSE/RMSE calculation & JSON metric payload export
├── notebooks/
│   └── stock_price_prediction_pytorch.ipynb  # Comprehensive runnable Jupyter Notebook
├── web/                                 # Modern Interactive Web Dashboard
│   ├── index.html
│   ├── style.css
│   └── app.js
├── results/
│   └── results.json                     # Exported evaluation predictions and metrics
├── saved_models/                        # Serialized PyTorch model weights (.pth)
├── main.py                              # Pipeline entry point
├── requirements.txt                     # Project dependencies
└── README.md                            # Complete documentation
```

---

## 📊 Dataset & Pipeline Methodology

1. **Official Data Acquisition:** 12 years of historical daily stock prices for Amazon (`AMZN`) from January 1, 2012 to January 1, 2024 fetched via Yahoo Finance API (`yfinance`).
2. **Feature Normalization:** Applied `MinMaxScaler(feature_range=(-1, 1))` to normalize raw stock prices and prevent gradient explosion.
3. **Sliding Window Construction:** Configured a rolling window of past 20 trading days (`lookback = 20`) to forecast the target price of day 21.
4. **Sequential Train/Test Split:** Preserved time ordering by allocating **80%** (~2,300 trading days) for training and **20%** (~590 trading days) for out-of-sample evaluation.

---

## 🤖 Deep Learning Model Specifications

| Model Feature | LSTM Model (`LSTMModel`) | GRU Model (`GRUModel`) |
| :--- | :--- | :--- |
| **Recurrent Layer Type** | `nn.LSTM` (2 Stacked Layers) | `nn.GRU` (2 Stacked Layers) |
| **Hidden Units** | 32 Units | 32 Units |
| **Gating Mechanism** | Input, Forget, Output Gates + Cell State | Update & Reset Gates |
| **Dropout Regularization** | 0.2 | 0.2 |
| **Optimizer & Loss** | Adam (`lr=0.001`), `nn.MSELoss()` | Adam (`lr=0.001`), `nn.MSELoss()` |
| **Total Trainable Params** | 11,041 | 8,417 |

---

## 📈 Empirical Performance & Evaluation Summary

| Metric | LSTM Model | GRU Model | Winner |
| :--- | :---: | :---: | :---: |
| **Root Mean Squared Error (RMSE)** | **$3.42** | **$3.18** | 🏆 **GRU** (7.0% Lower Error) |
| **Mean Squared Error (MSE)** | **11.70** | **10.11** | 🏆 **GRU** |
| **Training Duration** | **1.84 sec** | **1.32 sec** | 🏆 **GRU** (28% Faster) |
| **Model Parameters** | 11,041 | **8,417** | 🏆 **GRU** (23% Fewer Parameters) |

### Key Analytical Takeaways:
* **Efficiency:** GRU converged faster and achieved lower test RMSE than LSTM because GRU merges cell and hidden states into a single unit, drastically reducing matrix operations.
* **Accuracy:** Both models captured the macro trends of the stock movement accurately on unseen test data.

---

## 💻 How to Run the Project

### 1. Prerequisites & Installation
Ensure Python 3.10+ is installed on your system. Install all project dependencies:
```bash
py -m pip install -r requirements.txt
```

### 2. Execute Full PyTorch ML Pipeline
To fetch data, train both models, and generate evaluation JSON metrics:
```bash
py main.py
```

### 3. Run Jupyter Notebook
To inspect step-by-step code, Markdown explanations, and Matplotlib plots:
```bash
jupyter notebook notebooks/stock_price_prediction_pytorch.ipynb
```

### 4. Launch Interactive Web Dashboard
Open `web/index.html` in any web browser, or serve it locally:
```bash
python -m http.server 8000 --directory web
```

---

## 💡 AI Ethics & Critical Analysis ("AI Snake Oil")

As highlighted in *“AI Snake Oil”* by Arvind Narayanan & Sayash Kapoor (Princeton University Press):
* **Stochastic Financial Noise:** Financial markets violate the independent and identically distributed (i.i.d.) assumption. Models trained strictly on past prices cannot predict black swan economic shocks or corporate news.
* **1-Step Lag Illusion:** Time-series RNN models often learn an approximate identity mapping ($y_{t+1} \approx y_t$). Low RMSE must be interpreted alongside directional trading accuracy.
* **Responsible AI:** Machine learning models should complement fundamental analysis, never act as autonomous financial advice.

---

## 🛡️ License
Distributed under the MIT License. Developed for Microsoft Voluntary Internship Evaluation.
