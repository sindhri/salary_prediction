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
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img1.png" width="250">
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img2.png" width="500">

## 2.2 check salary = 0 situation
* The 0 salary has valid fields in other columns, so it does not look like it is an unpaid position
* There are only a small amount of them (n = 5)
* Remove salary = 0 rows
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img3.png" width="250">
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img4.png" width="250">

## 2.3 plot salary
* There are some outliers above the upper bound
* Salary has a nice normal distribution
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img5.png" width="900">

## 2.4 study salary outliers
* Salaries that are above the upper bounds have the job titles as 'CEO', 'CFO', 'CTO', 'VICE_PRESIDENT', 'SENIOR', 'MANAGER', which are not surprising. But it also has the job title 'JUNIOR'.
* Further examined the rows with salary above the upper bound and job title being 'JUNIOR', the degree field are all advanced degree, and the industry are all oil/web/finance. So they make sense
* Decide to keep the outliers.
<img src="https://github.com/sindhri/salary_prediction/blob/master/images/img12.png" width="500">
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
### 3.1 preprocessing
* combine the training features with salary
* drop duplicated jobId
* for both the training and test sets, convert the following variables to category 'companyId', 'jobType', 'degree', 'major', 'industry'
* remove the rows in training when salary = 0

