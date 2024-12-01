# House Sale Price Prediction 

A Streamlit-based application for predicting house sale prices based on the square footage of a house.

Access the app through: https://housepricepredictor-ayt8yjk593k34mohw7appus.streamlit.app/


## Features

- Upload a CSV file containing house size and price data.
- Visualize the uploaded data and its graph.
- Train a model using the uploaded dataset.
- View training results, including R² score and graphs of actual vs. predicted prices.
- Perform live predictions by inputting the square footage of a house.


## Requirements

- Python (latest version)
- Streamlit package


## Installation

1. Install the latest version of Python from [python.org](https://www.python.org/).
2. Open a command prompt and install Streamlit:
   ```bash
   pip install streamlit
  

## How To Run
1. Open a command prompt and navigate to the directory containing app.py
2. Run the following command:
   ```bash
   streamlit run <path-to-file>\app.py

4. The app will open in your web browser.


## Usage
### 1. Upload Data:
1. Upload a CSV file with houseSize and price columns.
2. Preview the data and view its graph.

### 2. Train the Model:
1. Click the Train Model button.
2. A success message ("Model trained successfully!") will be displayed.

### 3. View Training Results:
1. Click the Show Training Results button.
2. View the R² score and a graph of actual vs. predicted prices.

### 4. Live Prediction:
1. Input the square footage of a house.
2. View the predicted house price instantly.


## Example CSV file
```bash
houseSize,price
1200,200000
1500,250000
1800,300000
```


## Output
1. R² Score of the trained model.
2. Graph of actual vs. predicted house prices.
3. Real-time house price predictions.





