"""
Ankita Khadsare
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, plot_confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.svm import SVC


def get_variables():
	"""
	Function to get the independent and dependent variables
	:return: X,y
	"""

	# Read file into dataframe
	# df = pd.read_csv("sample_loan_applications.csv")	# uncomment this to run models on unbalanced data.
	df = pd.read_csv("sample_loan_applications_balanced.csv") # balanced data

	# Create a label encoder
	label_encoder = preprocessing.LabelEncoder()

	# Convert the string labels into numbers
	df = df.apply(lambda x: label_encoder.fit_transform(x))

	# Choose the independent and dependent variables
	X = df.loc[:, df.columns != 'TARGET'].values
	y = df.TARGET.values

	# Return independent and dependent variables
	return(X,y)


def run_naive_bayes():
	"""
	Naive Bayes
	:return: None
	"""

	# Get the independent and dependent variables
	X, y = get_variables()

	# Split the dataset in training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

	# Create a Gaussian classifier
	model = GaussianNB()

	# Train the model 
	model.fit(X_train, y_train)

	# Make prediction
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)

	# Make and print a confusion matrix
	conf_matr = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
	print(conf_matr)

	# Print the accuracy
	# print("Gaussian Naive Bayes model accuracy(in %) of training set:", accuracy_score(y_train, y_train_pred) * 100)
	print("Gaussian Naive Bayes model accuracy(in %) of test set:", accuracy_score(y_test, y_test_pred) * 100)

	# Print the classification report
	print(classification_report(y_test, y_test_pred))

	# Plot the data
	X_set, y_set = X_train[:, [7,8]], y_train
	for i, j in enumerate(np.unique(y_set)):
		plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], color = ListedColormap(('red', 'green'))(i), label = j, marker='.')
	plt.title('Training set')
	plt.xlabel('Client Income ($)')
	plt.ylabel('Amount of Loan ($)')
	plt.legend()
	# plt.savefig("Naive Bayes Output.png")
	plt.show()



def run_k_nearest_neighbors():
	"""
	K_Nearest_Neighbors
	:return: None
	"""
	# Get independent and dependent variables
	X,y = get_variables()

	# Split the dataset in training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

	# Create an error list
	error = []

	# Calculate the error rate for K values between 1 and 20
	for i in range(1, 20):
		knn = KNeighborsClassifier(n_neighbors = i)
		knn.fit(X_train, y_train)
		pred_i = knn.predict(X_test)
		error.append(np.mean(pred_i != y_test))

	# Plot the error rate
	plt.figure(figsize=(12, 6))
	plt.plot(range(1, 20), error, color = 'red', linestyle = 'dashed', marker = 'o', markerfacecolor = 'blue', markersize = 10)
	plt.title('Error Rate K Value')
	plt.xlabel('K Value')
	plt.ylabel('Mean Error')
	# plt.savefig('KNN Output.png')
	plt.show()

	# Create a KNN classifier
	model = KNeighborsClassifier(n_neighbors = 4)

	# Train the model 
	model.fit(X_train, y_train)

	# Make prediction
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)

	# Make and print a confusion matrix
	conf_matr = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
	print(conf_matr)

	# Print the accuracy
	# print("K-NN model accuracy(in %) of training set:", accuracy_score(y_train, y_train_pred) * 100)
	print("K-NN model accuracy(in %) of test set:", accuracy_score(y_test, y_test_pred) * 100)

	# Print the classification report
	print(classification_report(y_test, y_test_pred))


def run_logistic_regression():
	"""
	Logistic Regression
	:return: None
	"""
	# Get the independent and dependent variables
	X, y = get_variables()

	# Split the dataset in training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	# Build a logistic regression model
	model = LogisticRegression(max_iter=8000)

	# Train the model
	model.fit(X_train, y_train.ravel())

	# Make predictions
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)

	# Make and print a confusion matrix
	conf_matr = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
	print(conf_matr)

	# Print the accuracy
	# print("Accuracy of log_reg model on training set:", accuracy_score(y_train, y_train_pred) * 100)
	print("Accuracy of log_reg model on test set:", accuracy_score(y_test, y_test_pred) * 100)

	# Print the classification report
	print(classification_report(y_test, y_test_pred))


def run_support_vector_machines():
	"""
	Support Vector Machine
	:return: None
	"""

	# Get the independent and dependent variables
	X, y = get_variables()

	# Split the dataset in training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	# Build a SVM model
	model = SVC(kernel = 'rbf')

	# Train the model
	model.fit(X_train, y_train.ravel())

	# Make predictions
	y_train_pred = model.predict(X_train)
	y_test_pred = model.predict(X_test)

	# Make and print a confusion matrix
	conf_matr = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
	print(conf_matr)

	# Print the accuracy
	# print("Accuracy of SVM model on training set:", accuracy_score(y_train, y_train_pred) * 100)
	print("Accuracy of SVM model on test set:", accuracy_score(y_test, y_test_pred) * 100)

	# Print the classification report
	print(classification_report(y_test, y_test_pred))


def run_decision_tree():
	"""
	Decision Tree
	:return:
	"""

	# Get the independent and dependent variables
	X, y = get_variables()

	# Split the dataset in training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)
	print(X_train.shape)
	list_acc =[]
	x = []
	# for n in np.arange(0.01, 1, 0.05):
	# 	x.append("{:.2f}".format(n))
	for n in range(1):
		dTree = tree.DecisionTreeClassifier(criterion="gini", min_samples_split=0.05)
		dTree = dTree.fit(X_train, y_train)

		dTree_train_pred = dTree.predict(X_train)
		dTree_test_pred = dTree.predict(X_test)
		# print(dTree.feature_importances_)
		# print(dTree)

		# Make and print a confusion matrix
		conf_matr = pd.crosstab(y_test, dTree_test_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
		print(conf_matr)

		#Display confusion matrix plot
		disp = plot_confusion_matrix(dTree, X_test, y_test, display_labels=np.unique(y_test), cmap=plt.cm.Blues, normalize='true')
		disp.ax_.set_title("Confusion Matrix plot for Decision Tree")
		# plt.savefig("Decision Tree Confusion Matrix.png")
		plt.show()

		# Print the accuracy
		# print("Decision Tree model accuracy(in %) of training set:", accuracy_score(y_train, dTree_train_pred)*100)
		print("Decision Tree model accuracy(in %) of test set:", accuracy_score(y_test, dTree_test_pred)*100)
		list_acc.append(accuracy_score(y_test, dTree_test_pred)*100)
		metrics.plot_roc_curve(dTree, X_test, y_test)
		# plt.savefig("Decision Tree ROC.png")
		plt.show()


		# Print the classification report
		print(classification_report(y_test, dTree_test_pred))

		print("Decision Tree model average accuracy(in %) on Cross Validation", cross_val(X_train, y_train, model=dTree, num=1))
		break

	plt.show()

def run_random_forest():
	"""
	Random Forest
	:return: None
	"""

	# Get the independent and dependent variables
	X, y = get_variables()

	# Split the dataset in training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)

	rf = RandomForestClassifier(n_estimators=100, bootstrap=True, max_features='sqrt') #, class_weight={0: 5, 1: 1}) #, class_weight='balanced')

	rf = rf.fit(X_train, y_train)

	rf_train_pred = rf.predict(X_train)
	rf_test_pred = rf.predict(X_test)

	# Make and print a confusion matrix
	conf_matr = pd.crosstab(y_test, rf_test_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
	print(conf_matr)

	# Display confusion matrix plot
	disp = plot_confusion_matrix(rf, X_test, y_test, display_labels=np.unique(y_test), cmap=plt.cm.Blues, normalize='true')

	disp.ax_.set_title("Confusion Matrix plot for Random Forest")
	# plt.savefig("Random Forest Confusion Matrix.png")
	plt.show()

	# Print the accuracy
	# print("Random Forest model accuracy(in %) of training set:", accuracy_score(y_train, rf_train_pred) * 100)
	print("Random Forest model accuracy(in %) of test set:", accuracy_score(y_test, rf_test_pred) * 100)

	_1_probs = rf.predict_proba(X_test)[:, 1]
	_0_probs = rf.predict_proba(X_test)[:, 0]

	fpr, tpr, _ = metrics.roc_curve(y_test, _1_probs)
	plt.plot(fpr, tpr, linestyle='--', label='Class 1')
	fpr, tpr, _ = metrics.roc_curve(y_test, _0_probs)
	plt.plot(tpr, fpr, linestyle='--', label='Class 0')
	# plt.savefig("Random Forest ROC.png")
	plt.show()

	# Print the classification report
	print(classification_report(y_test, rf_test_pred))

	print("Decision Tree model average accuracy(in %) on Cross Validation", cross_val(X_train, y_train, model=rf, num=2))


# Function to calculate mean absolute error
def cross_val(X_train, y_train, model, num):
	# Applying k-Fold Cross Validation
	from sklearn.model_selection import cross_val_score
	accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=5)
	plt.plot(accuracies)
	plt.show()
	return accuracies.mean()


def main():
	print('-------------------------Naive Bayes------------------------')
	# Run Naive Bayes
	run_naive_bayes()

	print('-------------------------K-Nearest Neighbors------------------------')
	# Run K-Nearest Neighbors
	run_k_nearest_neighbors()

	print('-------------------------Logistic Regression------------------------')
	# Run Logistic Regression
	run_logistic_regression()

	print('-------------------------Support Vector Machine------------------------')
	# Run Support Vector Machine
	run_support_vector_machines()

	print('-------------------------Decision Tree------------------------')
	# Run Decision Tree
	run_decision_tree()

	print('-------------------------Random Forest------------------------')
	# Run Random Forest
	run_random_forest()


if __name__ == '__main__':
	main()