# Rossman Stores Sales Prediction

Our project attempted to apply various machine learning techniques to a real-world problem of predicting drug store sales. Rossmann, Germany’s second-largest drug store chain, has provided past sales information of 1115 Rossmann stores located across Germany. We preprocessed, feature-engineered the data, and examined different statistical / machine learning analysis for forecasting sales of each store. Then, we compared the methods’ predictive power by computing Root Mean Square Percentage Error (RMSPE). We found that Gradient Boosting model performed the best with a RMSPE score of 0.06 on the test data set.
For a more detailed analysis please refer to report.pdf
[Kaggle link to Rossmann Store Sales](https://www.kaggle.com/c/iiitb-ml-project-rossmann-store-sales).

## How to run the files
The file "Final_submission.py" consists of the pre-processing,model, predicting values on the test data and creating the pickle file.
Make sure the test.csv ,store.csv and train.csv present in the same folder as the python file.

Steps to run the code:
Run the Algorithm file on the datasets by entering the command -
      ----->python3 Final_submission.py
   The program will run for some amount of time and then produce a pickle file and submission.csv will store the predicted sales value.
