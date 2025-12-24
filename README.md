# Linear & Geometric Algebraic AI Text Detector

A transparent, interpretable AI text detection system based on **Linear Algebra** and **Geometric Topology**. Unlike black-box neural networks, this tool uses **Effective Rank** (intrinsic dimensionality) and **Total Variance** (semantic spread) to distinguish between human and AI-generated text.

## Features
* **Geometric Analysis:** Detects AI text using `Effective Rank` (a proxy for perplexity) and `Total Variance` (burstiness).
* **White-Box Interpretability:** Provides full mathematical reasoning for every detection result.
* **Ensemble Classifier:** Uses a weighted geometric scoring system (70% Rank / 30% Variance).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/hakamavicena/algeo-ai-text-detector.git](https://github.com/hakamavicena/algeo-ai-text-detector.git)
    cd algeo-ai-text-detector
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

Since the training process requires significant computational power to compute embeddings, it is recommended to run the training on Google Colab.

1.  **Download the Dataset:**
    Download the `AI_Human.csv` dataset from Kaggle:
    [https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)

2.  **Run Training on Colab:**
    * Open `threshold.py` (or your training notebook) in [Google Colab](https://colab.research.google.com/).
    * Upload the `AI_Human.csv` file to the Colab session.
    * Run the script. It will calculate the geometric thresholds (`tau_rank`, `tau_var`) and save the model.
    * Download the generated file: `detector_model_v2.pkl`.

3.  **Place Model in Project:**
    * Create a `models` folder in your project directory (if it doesn't exist):
        ```bash
        mkdir models
        ```
    * Move the downloaded `detector_model_v2.pkl` into the `models/` folder.

## Usage

1.  **Start the Web Application:**
    ```bash
    python app.py
    ```

2.  **Access the Interface:**
    Open your web browser and navigate to:
    `http://127.0.0.1:5000`

3.  **Test:**
    Paste text into the text area to see the geometric analysis and classification result.

## 📂 Project Structure
