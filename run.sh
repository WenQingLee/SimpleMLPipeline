#!/bin/bash
#!/usr/bin/python


# Welcome Message
echo "Welcome to a simple machine learning pipeline"

# Approach Selection
echo "Please indicate your approach ('Regression' or 'Classification'):"
read APPROACH

while [[ $APPROACH != 'Regression' ]] && [[ $APPROACH != 'Classification' ]]
do
	echo "Please indicate your approach ('Regression' or 'Classification'):"
	read APPROACH
done

# Model Selection
if [[ $APPROACH == 'Regression' ]]
then
	echo "Please select regression model ('GradientBoosting' or 'ElasticNet')"
	read MODEL_SELECT

	while [[ $MODEL_SELECT != 'GradientBoosting' ]] && [[ $MODEL_SELECT != 'ElasticNet' ]]
	do
		echo "Please select regression model ('GradientBoosting' or 'ElasticNet')"
		read MODEL_SELECT
	done

elif [[ $APPROACH == 'Classification' ]]
then
	echo "Please select classification model ('SupportVector_Bagging' or 'LogisticRegression_Bagging')"
	read MODEL_SELECT

	while [[ $MODEL_SELECT != 'SupportVector_Bagging' ]] && [[ $MODEL_SELECT != 'LogisticRegression_Bagging' ]]
	do
		echo "Please select classification model ('SupportVector_Bagging' or 'LogisticRegression_Bagging')"
		read MODEL_SELECT
	done
fi

# Feature Selection (No conditional statements to facilitate different data)
echo "A default list of features are selected. Please enter 'ok' if your accept or input the feature names to be used to estimate number of shares"
read FEATURE

if [[ $FEATURE == 'ok' ]]
then
	#python3 src/input_test.py $MODEL_SELECT
	python3 src/ml_model.py $MODEL_SELECT

else
	#python3 src/input_test.py $MODEL_SELECT $FEATURE
	python3 src/ml_model.py $MODEL_SELECT $FEATURE

fi
