
# Transaction Sentinel

# A Real-Time AI-Powered Fraud Detection System

Transaction Sentinel is a real-time, AI-powered system designed to combat financial fraud. The model analyzes transaction patterns, user behavior (location, purchase amount), and device data to instantly flag suspicious activity. This solution is trained on a dataset of transactions to learn what constitutes normal behavior for a user, enabling it to detect anomalies with high accuracy. This approach minimizes false positives, protecting both customers and businesses from financial loss.


# Key Features

  * **Real-time Analysis:** The system can analyze transaction data instantly through a web-based interface.
  * **AI-Powered Anomaly Detection:** Utilizes a **RandomForestClassifier** model to identify suspicious behavior that deviates from a user's normal patterns.
  * **High Accuracy:** Engineered to maintain a **low false-positive rate**, ensuring that legitimate transactions are not blocked unnecessarily.
  * **Improved Security:** Provides an additional layer of trust and security for customers by actively monitoring and preventing fraudulent activity.


# Technologies Used

  * **Backend:**

      * **Python:** The core language for the backend logic and machine learning model.
      * **Flask:** Used to create the web server and API for real-time predictions.
      * **pandas:** For data manipulation and processing.
      * **scikit-learn:** For model training, evaluation, and preprocessing.
      * **joblib:** For saving and loading the trained model and data encoder.

  * **Frontend:**

      * **HTML:** For the structure of the prediction dashboard.
      * **JavaScript:** For handling user input and communicating with the backend API.
      * **Tailwind CSS:** For styling the user interface.



# Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/manvi-chandra/Transaction-Sentinel.git
    cd Transaction-Sentinel
    ```

2.  **Install dependencies:**

    ```bash
    pip install pandas scikit-learn numpy flask flask-cors joblib
    ```

-----

# How to Use

Follow these steps in the specified order to run the project successfully:

1.  **Generate the Dataset:**
    Run the `datasetgenerator.py` script. This will create the synthetic dataset and save it as `transaction_data.csv`.

    ```bash
    python datasetgenerator.py
    ```

2.  **Train the AI Model:**
    Run `trainmodel.py`. This script will use the generated data to train the model and save it as `fraud_model.joblib` and `encoder.joblib`.

    ```bash
    python trainmodel.py
    ```

3.  **Start the Backend Server:**
    Run the `main.py` file to start the Flask server. This server will handle the prediction requests from the frontend.

    ```bash
    python main.py
    ```

4.  **Open the Frontend:**
    While the `main.py` server is running, open the `index.html` file in your web browser. You can now use the dashboard to enter transaction details and get a real-time fraud prediction.

-----

# File Descriptions

  * **`datasetgenerator.py`**: A script to create a synthetic dataset for the project, simulating both legitimate and fraudulent transactions.
  * **`trainmodel.py`**: Contains the logic to load the dataset, preprocess the data, train the fraud detection model, and save the trained model for later use.
  * **`main.py`**: The Flask application that acts as the backend server. It loads the trained model and exposes a `/predict` API endpoint to receive transaction data and return a prediction.
  * **`index.html`**: The frontend user interface for the project. It allows users to input transaction data and displays the prediction result received from the backend.
