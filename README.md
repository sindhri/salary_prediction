# Salary Prediction Summary
Predict salary using machine learning (regression, random forest)
<table>
  <tr>
    <th>Files</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td>/module/helpers.py</td>
    <td>tools built to facilitate EDA and preprocessing</td>
  </tr>
  <tr>
    <td>salary_EDA.ipynb</td>
    <td>EDA (Exploratory Data Analysis)</td>
  </tr>
  <tr>
    <td>salary_preprocessing.ipynb</td>
    <td>Data preprocess using scripts in modules/helpers</td>
  </tr>
    <tr>
    <td>salary_model.ipynb</td>
    <td>Machine Learning model building and turning</td>
  </tr>
</table>
<br>

# 1. DEFINE the problem
Salary is related to many factors such as major, job title and degree. In this project, we use multiple factors to predict salary. The factors include:
* CompanyID
* jobType
* degree
* major
* industry
* yearsExperience
* milesFromMetropolis

We use a metric MSE (Mean Squared Error) to assess the prediction accuracy. The lower MSE, the better the prediction.

# 2. DISCOVER
## 2.1 study the datasets
* 1 million records in the training set
* Numeric variables: yearsExperience, milesFromMetropolis, salary
* Categorical variables: jobId, companyId, jobType, degree, major, industry
* jobId is all unique, not included as a feature
* companyId has 63 unique values, can not easily visualize
* The rest of the categorical variables, jobType, degree, major and industry have a small amount of unique values and can visualize
* The minimum of salary is 0, need to check
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img1.png" width="500">
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img2.png" width="900">

## 2.2 check salary = 0 situation
* The 0 salary has valid fields in other columns, so it does not look like it is an unpaid position
* There are only a small amount of them (n = 5)
* Remove salary = 0 rows
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img3.png" width="900">
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img4.png" width="500">

## 2.3 plot salary
* There are some outliers above the upper bound
* Salary has a nice normal distribution
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img5.png" width="900">

## 2.4 study salary outliers
* Salaries that are above the upper bounds have the job titles as 'CEO', 'CFO', 'CTO', 'VICE_PRESIDENT', 'SENIOR', 'MANAGER', which are not surprising. But it also has the job title 'JUNIOR'.
* Further examined the rows with salary above the upper bound and job title being 'JUNIOR', the degree field are all advanced degree, and the industry are all oil/web/finance. So they make sense
* Decide to keep the outliers.
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img12.png" width="400">
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img13.png" width="900">

## 2.5 plot each feature in relation to salary
### jobType
* There were fairly equal amount of job types in the training set
* salary goes up in the order of 'JANITOR', 'JUNIOR', 'SENIOR', 'MANAGER', 'VICE_PRESIDENT', 'CFO', 'CTO', 'CEO'
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img6.png" width="900">

### degree
* There were more high school and none degrees than other categories
* The salaries in high school and none degrees are lower than other categories
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img7.png" width="900">

### major
* More than 50% of the case have none major.
* Cases that have a major are pretty evenly distributed across different majors.
* None major has a lower salary than any major
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img8.png" width="900">

### industry
* There were fairly equal amount of industry types in the training set
* salaries are the highest in 'FINANCE', 'OIL'
* salaries are the loest in 'EDUCATION', 'SERVICE'
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img9.png" width="900">

### yearsExperience
* Years of experience is fairly evenly distributed across the range of 0 to 24 years.
* There is a positive linear correlation between salary and years of experience
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img10.png" width="900">


### milesFromMetropolis
* Miles from metropolis is fairly evenly distributed across the range of 0 to 100 miles.
* There is a negative linear correlation between salary and miles from metropolis
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img11.png" width="900">

## 2.6 Encode categorical variables and plot feature correlation with salary
* Encode each categorical variables with the mean of the salary of that category
* Salary is positively related with encoded jobType, degree, major, industry and yearsExperience
* Salary is negatively related with milesFromMetropolis
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img14.png" width="900">

## 3. DEVELOP
### 3.1 preprocessing by scripts in modules/helpers.py
* combine the training features with salary
* drop rows with duplicated jobId
* for both the training and test sets, convert the following variables to category 'companyId', 'jobType', 'degree', 'major', 'industry'
* remove the rows in training when salary = 0
* transform categorical variables to be the mean salary of each category and use them as part of the feature group: feature_transform
* encode categorical variables and use them as part of the feature group: feature_encode

### 3.2 develop the model
* Establish the metrics, will use Mean Square Error (MSE) as the metrics to determine prediction accuracy
* Test the baseline prediction, tried using the following transformed categorical columns as the prediction: industry, major, degree, jobType. JobType generated the smallest MSE = 964.1529
<table>
  <tr>
    <th>baseline models</th>
    <th>MSE</th>
  </tr>
  <tr>
    <td>mean salary for each industry</td>
    <td>1367.5539</td>
  </tr>
  <tr>
    <td>mean salary for each major</td>
    <td>1284.3599</td>
  </tr>
  <tr>
    <td>mean salary for each degree</td>
    <td>1257.9450</td>
  </tr>
    <tr>
    <td>mean salary for each jobType</td>
    <td>964.1529</td>
  </tr>
</table>

* Several Machine Learning Models are tried with different feature combinations
* note: transformed means categorical features transformed to be the mean salary of each category.
* note: encoded means categorical features were coded with arbitary numeric numbers.
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img18.png" width="900">
* note: also included an option of whether to use scaler (normalize all the numeric variables).

* Observations:
* Different models generated similar MSE, except linear regression did really poorly with encoded features.
* The model with the lowest MSE is GradientBoosting with transformed features and scaled.
<table>
  <tr>
    <th>Features</th>
    <th>ML models</th>
    <th>MSE</th>
  </tr>
  <tr>
    <th>Numeric + transformed categorical not scaled</th>
    <td>Linear Regression</td>
    <td>399.7641</td>
  </tr>
  <tr>
    <th></th>
    <td>Random Forest</td>
    <td>365.8091</td>
  </tr>
  <tr>
    <th></th>
    <td>GradientBoosting</td>
    <td>364.1131</td>
  </tr>
  <tr>
    <th>Numeric + transformed categorical scaled</th>
    <td>Linear Regression</td>
    <td>399.7641</td>
  </tr>
  <tr>
    <th></th>
    <td>Random Forest</td>
    <td>365.7466</td>
  </tr>
  <tr>
    <th></th>
    <td>GradientBoosting</td>
    <td>364.1131</td>
  </tr>
  <tr>
    <th>Numeric + encoded categorical not scaled</th>
    <td>Linear Regression</td>
    <td>925.0988</td>
  </tr>
  <tr>
    <th></th>
    <td>Random Forest</td>
    <td>372.4568</td>
  </tr>
  <tr>
    <th></th>
    <td>GradientBoosting</td>
    <td>379.0314</td>
  </tr>
  <tr>
    <th>Numeric + encoded categorical scaled</th>
    <td>Linear Regression</td>
    <td>925.0988</td>
  </tr>
  <tr>
    <th></th>
    <td>Random Forest</td>
    <td>372.5084</td>
  </tr>
  <tr>
    <th></th>
    <td>GradientBoosting</td>
    <td>379.0315</td>
  </tr>
</table>

### 3.3 choose the best model and buid pipeline
* GradientBoosting with transformed and scaled features provided the best results (lowest MSE)
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img15.png" width="900">


### 3.4 Train the model on the whole dataset

## 4. DEPLOY
### Apply the model on the test data
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img16.png" width="900">

### Feature importance
* jobType_transformed has the greatest contribution. It is consistent with our baseline analysis where using the mean average for each jobType gave us the best baseline results (compared to using degree, major and industry)
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img17.png" width="900">

### Improve the model
* The model can be improved by removing some colinearity in the data for linear regression
