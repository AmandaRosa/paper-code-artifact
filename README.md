# Multi-Fault Diagnosis Using Lightweight Machine Learning Techniques for Rotordynamics Analysis in Multi-Domain

This repository contains an embedded system for real-time monitoring and fault diagnosis of rotating machinery, specifically designed for edge devices such as the Raspberry Pi. The system employs advanced signal processing techniques and machine learning ensembles to detect common mechanical faults, including unbalance and misalignments.

## 🚀 Key Features

- **Real-time Fault Detection:** Monitors vibration signals to classify machinery states into:
  - `Normal`
  - `Unbalance`
  - `Horizontal Misalignment`
  - `Vertical Misalignment`
  - `Rubbing`
  - `Crack`

  - *COMFAULDA dataset can be found at: https://ieee-dataport.org/documents/composed-fault-dataset-comfaulda*
  - *ROSS signal generated can be found at: https://drive.google.com/drive/folders/1OUiW2tr4lx8QIQr9yf4KlgiegyNANZy6?usp=sharing*
    
- **Advanced Signal Processing:**
  - **Time Domain:** RMS, Standard Deviation, Kurtosis, etc.
  - **Frequency Domain:** Power Spectral Density (PSD), FFT peak analysis.
  - **Time-Frequency Domain:** STFT and Spectrogram analysis with image descriptors (HOG and LBP).
  - **Dimensionality Reduction:** Principal Component Analysis (PCA) for feature selection/reduction.
  - **Preprocessing:** Adaptive thresholding using Hilbert Transform and Butterworth low-pass filtering.
- **Machine Learning Ensemble:** Utilizes a voting-based ensemble of pre-trained models including **Random Forest**, **Multi-Layer Perceptron (MLP)**, and **Support Vector Machines (SVM)**.
- **Efficient Edge Deployment:**
  - MQTT-based data ingestion for decoupled communication.
  - Lightweight SQLite database for local result storage.
  - Resource monitoring (CPU/RAM usage) logged for performance analysis.
- **REST API:** Built with FastAPI for easy retrieval of classification results.

## 📁 Repository Structure

```text
.
├── data/                   # Experimental vibration data (Normal, Unbalance, Misalignment)
└── raspberry/              # Core application for edge devices
    ├── main.py             # Entry point (FastAPI + MQTT Subscriber)
    ├── framework.py        # Core processing pipeline & classification logic
    ├── subscriber.py       # MQTT client for data ingestion
    ├── models/             # Pre-trained Scikit-learn models (.pkl)
    │   ├── T-F-models/     # Time-Frequency domain models
    │   └── TF-models/      # Time and Frequency domain models
    ├── utils/              # Signal processing & feature extraction utilities
    ├── raspberry_database.db # Local SQLite database for results
    └── raspberry_requirements.txt # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- An MQTT Broker (e.g., [Mosquitto](https://mosquitto.org/))

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required dependencies:
   ```bash
   pip install -r raspberry/raspberry_requirements.txt
   ```

## 💻 Usage

1. **Start the MQTT Broker:** Ensure your MQTT broker is running on `localhost` (or update `broker_address` in `raspberry/subscriber.py`).

2. **Run the Application:**
   ```bash
   cd raspberry
   python main.py
   ```
   The system will start the FastAPI server and the MQTT subscriber thread.

3. **Data Ingestion:**
   The system listens for vibration data on the MQTT topic `test/topic1`. Data should be sent in JSON format.

4. **Retrieve Results:**
   You can access the results via the REST API:
   - `GET /get_results`: Returns all classification results stored in the local database.

## 📊 Experimental Data

The `data/` directory contains sample vibration signals used for training and testing the models. The data is organized by fault type and severity (e.g., `Horizontal_Misalignment_0_5mm`, `Unbalance_10g`). These files are in `.txt` or `.csv` format, representing raw acceleration data.

## 📄 Citation

If you use this code or dataset in your research, please cite:

> Jorge, Amanda Rosa Ferreira. *Multi-Fault Diagnosis Using Lightweight Machine Learning Techniques for Rotordynamics Analysis in Multi-Domain*. Submitted to **APPLIED ARTIFICIAL INTELLIGENCE**, 2026.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
