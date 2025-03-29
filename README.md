# Boston Housing Linear Regression

This project demonstrates a linear regression model built using scikit-learn to predict the median value of owner-occupied homes (`medv`) in the Boston Housing dataset. The code includes steps for reading data, preprocessing, model training with train-test splitting, model persistence using Joblib, evaluation using mean squared error and R² metrics, and visualization of the model's performance.

---

## Dataset

The dataset used in this project is a CSV file (`BostonHousing.csv`) with the following columns:

- **crim:** per capita crime rate by town  
- **zn:** proportion of residential land zoned for lots over 25,000 sq.ft.  
- **indus:** proportion of non-retail business acres per town  
- **chas:** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)  
- **nox:** nitric oxides concentration (parts per 10 million)  
- **rm:** average number of rooms per dwelling  
- **age:** proportion of owner-occupied units built prior to 1940  
- **dis:** weighted distances to five Boston employment centres  
- **rad:** index of accessibility to radial highways  
- **tax:** full-value property-tax rate per \$10,000  
- **ptratio:** pupil-teacher ratio by town  
- **b:** 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town  
- **lstat:** % lower status of the population  
- **medv:** Median value of owner-occupied homes in \$1000's (target variable)

An example of the data:

| crim   | zn  | indus | chas | nox   | rm    | age  | dis   | rad | tax | ptratio | b     | lstat | medv |
|--------|-----|-------|------|-------|-------|------|-------|-----|-----|---------|-------|-------|------|
| 0.00632| 18  | 2.31  | 0    | 0.538 | 6.575 | 65.2 | 4.09  | 1   | 296 | 15.3    | 396.9 | 4.98  | 24   |
| 0.02731| 0   | 7.07  | 0    | 0.469 | 6.421 | 78.9 | 4.9671| 2   | 242 | 17.8    | 396.9 | 9.14  | 21.6 |
| 0.02729| 0   | 7.07  | 0    | 0.469 | 7.185 | 61.1 | 4.9671| 2   | 242 | 17.8    | 392.83| 4.03  | 34.7 |

---

## Project Structure

- **model2.py:**  
  The main Python script that:
  - Reads the dataset.
  - Separates features (`x`) and target (`y`).
  - Splits the data into training and testing sets using `train_test_split`.
  - Trains a linear regression model on the training data (or loads a previously saved model using Joblib).
  - Evaluates the model using mean squared error (MSE) and R² scores.
  - Visualizes the predictions using a scatter plot.
  
- **BostonHousing.csv:**  
  The CSV file containing the Boston Housing dataset.

- **linear_model.pkl:**  
  The file where the trained model is saved using Joblib (saved in the specified path).

---

## Requirements

- **Python 3.x**
- **Pandas**  
  `pip install pandas`
- **scikit-learn**  
  `pip install scikit-learn`
- **Matplotlib**  
  `pip install matplotlib`
- **Joblib** (often installed with scikit-learn)  
  `pip install joblib`

---

## How to Run

1. **Clone/Download the Repository:**  
   Ensure that you have both `model2.py` and `BostonHousing.csv` in your project directory.

2. **Configure Your Environment:**  
   Make sure that you have installed all required packages (Pandas, scikit-learn, matplotlib, joblib). You can install them using pip.

3. **Run the Script:**  
   Open a terminal (or use PyCharm's run configuration) and execute:
   ```bash
   python model2.py
   ```
   The script will either load the saved model if it exists or train a new linear regression model. It will then evaluate the model and display the results along with a scatter plot of the training predictions.

---

## Results

The evaluation of the model on the Boston Housing dataset produces metrics like:
```
              method    mse_train  r2_train   mse_test   r2_test
0  Linear Regression   21.649377  0.733733  23.616994  0.755503
```
- **MSE (Mean Squared Error):** Measures the average squared difference between the predicted and actual values. Lower values indicate better performance.
- **R² Score:** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. Values closer to 1 indicate a better fit.

---

## Visualization

The script generates a scatter plot that compares:
- **Actual Training Values (`ytrain`)** on one axis.
- **Predicted Training Values (`y_lr_train_pred`)** on the other axis.

This visualization helps assess how well the model's predictions align with the actual values.

---
