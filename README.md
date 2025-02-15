# CODEALPHA-TASK-2
### UNEMPLOYMENT ANALYSIS WITH PYTHON ###
This code, executed in a Jupyter Notebook, aims to predict the unemployment rate using a linear regression model. The dataset contains information on the estimated unemployment rate, number of employed people, and labor participation rate. The code performs several key tasks including data cleaning, visualization, training a linear regression model, evaluating the model, and making future predictions.

 Step 1: **Importing Required Libraries**
The necessary libraries for data manipulation, machine learning, and visualization are imported:
- **pandas** is used to read and manage data.
- **numpy** helps with numerical operations.
- **matplotlib.pyplot** and **seaborn** are utilized for creating graphs such as line plots and heatmaps.
- **train_test_split** from **sklearn.model_selection** is used to split the dataset into training and testing sets.
- **LinearRegression** from **sklearn.linear_model** is the machine learning model used for prediction.
- **mean_absolute_error**, **mean_squared_error**, and **r2_score** from **sklearn.metrics** are used to evaluate the model's performance.

 Step 2: **Reading the Dataset**
The dataset is loaded into a pandas DataFrame using `read_csv()`. After reading the data, the column names are stripped of leading or trailing spaces to ensure proper formatting.

 Step 3: **Handling Missing Data**
Next, the code checks for missing values in the dataset. Any missing values in numerical columns are replaced with the median of that column using the `fillna()` function. This helps prevent issues that could arise from missing data when building the model.

 Step 4: **Date Conversion**
The 'Date' column, which is in string format, is converted into a pandas datetime object using `pd.to_datetime()`. This allows for proper handling of dates, which is important for time series analysis. Any rows with invalid dates are removed using `dropna()`, ensuring that only valid date entries remain in the dataset.

 Step 5: **Visualization**
Before proceeding with modeling, the code generates two visualizations:
1. **Line Plot**: A line plot is created to visualize the unemployment rate over time. This helps in understanding how the unemployment rate has changed, providing context for the modeling process.
2. **Correlation Heatmap**: A heatmap is created to show the correlation between the unemployment rate, the number of employed individuals, and the labor participation rate. This helps to identify relationships between variables, which can inform the feature selection process for the model.

 Step 6: **Feature Selection**
The features used for prediction are the number of employed individuals and the labor participation rate, while the target variable is the unemployment rate. The dataset is divided into two parts: the independent variables (X) and the dependent variable (y). This is done so that the model can learn to predict the unemployment rate based on the other variables.

 Step 7: **Splitting the Data**
The data is split into training and testing sets using the `train_test_split()` function. Typically, the training set is used to train the model, and the test set is used to evaluate its performance.

 Step 8: **Training the Model**
A linear regression model is created and trained on the training data using the `fit()` function. This model learns the relationship between the independent variables and the target variable.

 Step 9: **Evaluating the Model**
Once the model is trained, predictions are made on the test set. The model's performance is evaluated using three metrics:
- **Mean Absolute Error (MAE)**: This measures the average magnitude of errors between predicted and actual values.
- **Mean Squared Error (MSE)**: Similar to MAE but gives more weight to larger errors.
- **R² Score**: This shows how well the model explains the variance in the target variable, with a higher value indicating a better fit.

 Step 10: **Visualization of Predictions**
A scatter plot is created to visualize the actual vs. predicted unemployment rates. A red dashed line represents a perfect prediction (where actual equals predicted), and points closer to this line indicate better predictions.

 Step 11: **Making Future Predictions**
Finally, the model is used to predict the unemployment rate for future data based on given values for the number of employed individuals and the labor participation rate. The predicted value is printed as the output.

In summary, this code loads, cleans, and visualizes unemployment data, builds a linear regression model to predict the unemployment rate, evaluates the model's performance, and makes predictions for future scenarios.
***OUTPUT***:![Image](https://github.com/user-attachments/assets/d42b7ce8-bda8-44f4-b0f4-8bdc9b3c473d)
