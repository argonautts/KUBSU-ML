# Machine learning

## Numpy
1. Create a vector (one-dimensional array) of size 10 filled with
with units.
2. Create a vector (one-dimensional array) with values from 10 to 49
3. Create a vector (one-dimensional array) of size 10. Fill it
with random values. Find indices of non-zero elements
4. Create a 3x3 matrix (two-dimensional array) with values from 0 to 8.
5. Create an 8x8 matrix and fill it in staggered order using the
tile function
6. Create a vector (one-dimensional array) of size 10. Fill it
with random values. Replace the minimum element with zero
7. Create a vector (one-dimensional array) of size 100. Fill it
with random values. Find the most frequent value in the array
8. Create a matrix. Subtract the average of each row in the matrix
9. Create a matrix. Swap two rows in the matrix
10. Create a vector (one-dimensional array) of size 10. Fill it
with random values. Find the n largest values in the vector
11. Create a 5x5 matrix with values in the rows from 1 to 5.
12. Create two 4x4 and 4x4 matrices. Find their product. Find
diagonal elements of the product of matrices

## Pandas

Let us consider a dataset with data collected through a survey of students
of a high school math course in Portugal (ages ranging from 15 to 22).
("math_students.csv"). The target variable is the student's final grade for the
course.
Detailed transcription of the attributes:
- school - type of school ("GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)
- sex - gender ("F" - female or "M" - male)
- age - age (from 15 to 22)
- address - where the student is from ("U" - urban or "R" - rural)
- famsize - family size ("LE3" - less than or equal to 3 or "GT3" - greater than 3)
- Pstatus - what is the parents' relationship ("T" - living together "A" - separated)
- Medu - mother's education (0 - no education, 1 - primary education (4 grades), 2 - from 5 to 9 grades, 3 - secondary education).
9 grades, 3 - secondary or 4 - higher education)
- Fedu - father's education (0 - none, 1 - primary education (4 grades), 2 - 5 to 9 grades, 3 - secondary or 4 - higher)
grades, 3 - secondary or 4 - higher)
- Mjob - mother's job ("teacher", "health" care related, civil "services" (e.g. administrative or
police), "at_home" or "other")
- Fjob - father's job ("teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other").
police), "at_home" or "other")
- reason - reason for choosing a school (close to home - "home", school reputation - "reputation",
preference for certain subjects - "course" or "other")
- guardian - guardian ("mother", "father" or "other")
- traveltime - time from home to school (1 - less than 15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - more than 1 hour).
hour, or 4 - more than 1 hour)
- studytime - number of hours of training per week (1 - less than 2 hours, 2 - from 2 to 5 hours, 3 - from 5 to 10 hours, or 4 - more than 10 hours).
from 5 to 10 hours, or 4 - more than 10 hours)
- failures - number of previously failed subjects (n if 1 <= n < 3, else 4)
- schoolup - extra classes (yes or no)
- famsup - help from family in completing assignments (yes or no)
- paid - additional paid classes (yes or no)
- activities - extracurricular activities (yes or no)
- nursery - attended kindergarten (yes or no)
- higher - desire for higher education (yes or no)
- internet - home internet (yes or no)
- romantic - in a romantic relationship (yes or no)
- famrel - how good is the relationship in the family (from 1 - very bad to 5 - excellent)
- freetime - has free time after school (from 1 - very little to 5 - a lot)
- goout - goes out with friends (from 1 - rarely to 5 - very often)
- Dalc - drinking alcohol on weekdays (from 1 - very rarely to 5 - very often)
- Walc - weekend alcohol use (from 1 - very rare to 5 - very frequent)
- health - current state of health (from 1 - very bad to 5 - very good)
- absences - number of school absences (from 0 to 93)
- G1 - grade for the first semester (from 0 to 20)
- G2 - grade for the second semester (from 0 to 20)
- G3 - final grade (0 to 20)

1. Load data (read_csv())
2. Print the first and last 10 rows of the table (head(), tail())
3. Print the number of objects and their characteristics
4. Print the names of all columns
5. Whether there are any gaps in the data
6. Output statistics on the values of features
7. Output a more detailed description of the feature values (number of non-empty values, column types and volume). values, column types and memory size)
8. What values does one of the attributes (e.g. guardian) take? (what values, how many unique values, how many values of each type)
9. Output only those students who have a guardian who is a mother and who works as a teacher or works from home:
10. Create an "alc" trait that reflects total alcohol consumption during the week
11. Output new size, new columns
12. What was the most frequent reason for school choice? Provide the corresponding value of the trait.
13. Find the number of students whose parents have no education.
14. Find the minimum age of a student at Mousinho da Silveira School.
15. Find the number of students who have an odd number of absences.
16. Find the difference between the average final grades of students who are and are not  who are in a romantic relationship. For your answer, give a number rounded  to two significant digits after the decimal point.

## Matplotlib

Consider a dataset that contains data on passengers from the "Titanic" (titanic.csv).
0. Perform an initial analysis of the data. Provide the following
information about the dataset:

- description of the dataset, explanations to better understand the nature of the data.
- A description of each feature and its type.
- the shape of the data set: number of items in the set, number of
attributes, number of missing values, average value of individual attributes, maximum and minimum values.
of individual features, maximum and minimum values of individual features.
values of individual attributes and other indicators.
- presence/absence of empty values; if there are empty values, it is necessary to get rid of them.
values, it is necessary to get rid of them

1. Find the number of observations for each value of the target
Survived variable and apply the plot method to the obtained data in order to plot a bar chart.
plot a bar chart.
2. Construct the same chart using a logarithmic scale.
3. Construct two histograms of the Pclass values - one for the
survivors (Survived equals 1) and one for non-survivors (Survived equals 0).
4. Select the value of the density argument so that the vertical line of the graph does not contain the number of observations.
of the graph is the density of the distribution rather than the number of observations.
The number of bins should be equal to 20 for both histograms and the coefficient alpha
equal to 0.5 so that the histograms are semi-transparent and do not obscure each other.
each other.
5. Create a legend with two values: " Survived 0" and " Survived 1".
The histogram of normal transactions should be gray, and the histogram of
and the histogram of fraudulent transactions should be red. The name of the horizontal axis is " Survived ".
6. Similarly depict the histograms

## Linear Regreassion
1. Load the "Boston Housing" dataset from the scikit library-learn (load_boston).
2. examine the structure and content of the dataset.
3. Perform preliminary analysis of the data, including checking for
the presence of missing values, outliers, and correlation between the
variables.
4. Split the data into training and test samples in a 70/30 ratio.
70/30 ratio.
5. Train a linear regression model on the training sample.
6. Evaluate the quality of the model on the test sample with the help of such
metrics such as the mean absolute error (MAE), mean square error
(MSE), root mean square error (RMSE), and coefficient of
of determination (R^2).
7. Visualize the prediction results by comparing the original
real estate price values with the predicted values.
8. Perform a cross-validation of the model and evaluate its quality using the
MSE, RMSE, and R^2 metrics.
9. To try to improve the quality of the model by selecting the most
significant variables, using regularization or other methods.
10. Draw conclusions about the quality of the model and its applicability to the
predicting housing prices in Boston.

## Classification

1. Load the dataset "Iris" from the scikitlearn library (load_iris).
2. study the structure and content of the dataset.
3. Perform preliminary analysis of the data, including checking for missing values and correlation between variables.
missing values and correlations between variables.
4. Divide the data into training and test samples in the ratio of
70/30.
5. Train a logistic regression model (LogisticRegression) on the training sample.
training sample.
6. Evaluate the quality of the model on the test sample using metrics
such as accuracy, completeness, F1-measure and error matrix (accuracy_score,
precision_score, recall_score, f1_score, confusion_matrix)
7. Visualize the prediction results by comparing the original
values of flower classes and the predicted values.
8. Perform cross-validation of the model and evaluate its quality using the
metrics of accuracy, completeness, F1-measure and error matrix.
9. Try to improve the quality of the model by varying the parameters of the
model or using other models (e.g., k-nearest neighbors
(KNeighborsClassifier), support vector method (SVMClassifier), or random forest (RandomForest).
Random Forest (RandomForestClassifier)).
10. Draw conclusions about the quality of the model and its applicability to the
classification of Iris flower types.

## Clustering

Consider a dataset with data about supermarket customers, including their gender, age, income, and spending estimates ("Customers.csv"). Load the data (read_csv())
1. Output the first and last 10 rows of the table (head(), tail()).
2. Output statistics by feature values.
3. Output a more detailed description of the attribute values (number of non-empty values, column types and memory space). values, column types and memory size).
4. Visualize the gender of the client. Construct a bar chart and a pie chart to show the gender distribution on the dataset.
5. Visualize age distribution. Construct a histogram, to determine the the age distribution of customers. Also construct a block diagram (box and whiskers).
6. Analyze the annual income of customers. Construct a histogram, then examine this data using a density plot.
7. Analyze an estimate of customer expenditures. Analyze expenses using various graphs.
8. Using K-means algorithm by varying its various parameters to determine the optimum number of clusters
9. Visualize the clustering results

## PyTorch

1. Download the MNIST dataset, which includes handwritten
digits from 0 to 9.
2. Preprocess the data to bring it to the desired
format and scale.
3. Create a neural network model using PyTorch. The model should contain multiple layers, including hidden layers.
4. train the model on a training dataset using the loss function and optimizer from PyTorch.
5. Evaluate the quality of the model on the test dataset.
6. Change the model parameters (e.g., number of hidden layers, number of neurons in the layers) and compare their effect on the quality of of recognition.
7. Change the activation function in the neural network (e.g., ReLU, sigmoid) and compare their effect on the recognition quality.



## TensorFlow

1. Select a dataset to train the image classification model. Well-known datasets such as CIFAR-10, MNIST can be used.
2. Configure the model architecture using TensorFlow. You can choose any model, such as convolutional neural networks (CNNs), or use an off-the-shelf architecture such as VGG, ResNet or Inception and apply it to the dataset.
3. Divide the data set into training and test sets. It is recommended to use a proportion of 80% training data and 20% test data.
4. Set up the model training process, including the choice of loss function, optimiser and training hyperparameters. You can use a loss function such as categorical cross-entropy and an optimiser such as stochastic gradient descent (SGD) or Adam.
5. Train the model on training data and evaluate its performance using metrics such as accuracy or confusion matrix on a test dataset.
6. Analyse the results, draw conclusions about the performance of the model and possible ways to improve it. Experiment with different architectures, hyperparameters and regularisation methods to improve classification accuracy.
7. Bonus task: implement the ability to validate the model on real images that it has not seen during training. Test the model on a small set of real images and evaluate its performance.

---

## Individual assignment (regression)

**Air Pollution Prediction**

The dataset BeijingPM20100101_20151231.csv contains data on air pollution in Beijing, China from 1 January 2010 to 31 December 2015. The data were collected from
through several air quality monitoring stations placed throughout the city. Each record in the dataset contains air quality information for a specific
date and time.

1. Dataset Loading.
2. Exploring the dimensionality of the dataset and variable types.
3. Checking for missing values and deciding whether to process them (filling or deleting missing values).
4. Analyse the distribution of variables using descriptive statistics (mean, median, standard deviation).
5. Examination of the correlation matrix between all variables to identify possible dependencies.
6. Visualisation of the correlation matrix using heat map to visualise the strength and direction of relationships between variables.
7. Study the distribution of the target variable (air pollution) using histogram.
8. Construction of boxplots to detect outliers and anomalous values in numerical variables.
9. Convert categorical attributes (if any) to numeric attributes by creating dummy variables.
10. Splitting the dataset into training and test samples for subsequent training and estimation of the regression model.
11. Applying a method of scaling (e.g. standardisation) of numerical features to improve model performance.
12. Selecting a regression model (e.g., linear regression or random forest) and training the model on the training data.
13. Estimating the model on test data using regression metrics such as coefficient of determination (R^2) or mean square error (MSE).
14. Interpretation of the obtained results and analysing the significance of the features using the model coefficients.
15. Predicting air pollution consumption for new observations based on a trained regression model.
16. Visualising the relationship between air pollution variables and other factors using scatter plots.
17. Applying data transformations (e.g. logarithmisation) to improve the linearity of the relationship between variables.
18. Assessing the significance of selected features using statistical tests (e.g. t-test).
19. Creating new features based on existing features (e.g., combination of two variables).
20. Checking the model for overfitting and selecting the optimal number of features using cross-validation.


## Individual assignment (classification)

**Predict whether a person will agree to open a bank deposit**

Description: You need to build a classification model to predict whether a customer will sign up for a bank deposit based on various factors provided in the Bank Marketing dataset.

1. Loading and reviewing the dataset:
   - Load the dataset and view the first 5 rows;
   - Viewing general information about the data (number of rows, data types, presence of missing values, etc.);
   - Examining the statistics of the data (mean, standard deviation, maximum and minimum values);
   - Examining the distribution of the target variable (number of positive and negative classes).
2. Data pre-processing:
   - Processing missing values (filling in or deleting missing data);
   - Convert categorical attributes to numeric values using encoding, e.g. using the One-Hot Encoding method;
   - Scaling of numeric variables (normalisation or standardisation) (e.g. using the StandardScaler method)
3. Data visualisation:
   - Constructing histograms for numerical variables to examine their distribution;
   - Constructing bar charts for categorical variables to examine their distribution;
   - Constructing a correlation matrix to examine the relationship between variables.
4. Preparation of data for modelling:
   - Separation of data into training and test samples;
   - Uniform distribution of classes in training and test samples (application of methods for handling unbalanced data).
5. Building a classification model:
   - Selecting appropriate machine learning algorithms for classification (e.g., logistic regression, random forest, gradient bousting, etc.);
   - Training the model on the training sample;
   - Assessing the quality of the model using metrics (accuracy, completeness, F1-measure, ROC-curve, etc.);
   - Selection of model hyperparameters to achieve better performance.

Repeat step 5 for other classification algorithms and compare them with each other.