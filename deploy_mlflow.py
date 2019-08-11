"""
mlFlow can separate into tracking, projects and models

Tracking: Tracking allows you to create an extensive logging framework around your model. You can define custome metrics so that after a run you 
can compare the output to previous runs.

Projects: This feature allow you to create a pipeline if you so desire. This feature uses its own template to define how you want to run the model
on a cloud enviroment.

Models: An mlFlow Model is a standard format for packaging machine learning models that can be used in a variety of downstream tools.
"""

# Import the Libraries we will need in mlFlow
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
warnings.filterwarnings('ignore')

def eval_metric(actual, predict):
	rmse = np.sqrt(mean_absolute_error(actual, predict))
	mae = mean_absolute_error(actual, predict)
	r2 = r2_score(actual, predict)
	return rmse, mae, r2

if __name__ == '__main__':
	data = load_iris()

	# BUild dataframe
	data_df = pd.DataFrame(
	                np.c_[data['data'], data['target']],
	                columns=data['feature_names']+['target'])

	# Separate to train and test datasets
	train, test = train_test_split(data_df)

	X_train = train[data['feature_names']]
	y_train = train['target']
	X_test = test[data['feature_names']]
	y_test = test['target']

	# Set parameters to tune
	tuned_parameters = [
	    {'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 10},
	    {'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 10},
	    {'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 30},
	    {'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 50},
	]

	# Run the mlflow tracking
	for param in tuned_parameters:
		with mlflow.start_run():
			lr = LogisticRegression(max_iter=param['max_iter'], penalty=param['penalty'])
			lr.fit(X_train, y_train)
			y_predict = lr.predict(X_test)

			(rmse, mae, r2) = eval_metric(y_test, y_predict)

			# Metric will appear on the web
			mlflow.log_metric('mae', mae)
			mlflow.log_metric('rmse', rmse)
			mlflow.log_metric('r2', r2)
			# Parameters will appear on the web
			mlflow.log_param('penalty', param['penalty'])
			mlflow.log_param('max_iter', param['max_iter'])

			# Log a scikit-learn model as an MLflow artifact for the current run.
			mlflow.sklearn.log_model(lr, 'model')