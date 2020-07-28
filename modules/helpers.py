import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

#read in an csv file and display the columns and head
def read_in_dataset(fname, verbose = False):
    data = pd.read_csv(fname)
    if verbose:
        print('\n{0:*^80}'.format('Reading in the {0} dataset'.format(fname)))
        print('\nit has {0} rows and {1} columns'.format(*data.shape))
        print('\n{0:*^80}'.format('it has the following columns\n'))
        print(data.columns)
        print('\n{0:*^80}\n'.format('the first 5 rows looks like this'))
        print(data.head())
    return data

#if a value of a categorical variable in the testing set does not exist in the training set, set the value to be the mode of the training set of that categorical variable
def fill_extra_categories_with_train_mode(df, colname, train):
    extra_values = set(df[colname]) - set(train[colname])
    replacement = train[colname].mode()[0]
    
    for value in extra_values:
        bool1 = df[colname] == value
        print('\nreplaced {0} {1} with {2}'.format(sum(bool1), value, replacement))
        df.loc[bool1,colname] = replacement
    return df

#remove columns that are not going to the models
def remove_extra_columns(df, colnames):
    df = df.drop(columns = colnames)
    return df

#fill in the empty cells for column1 base on the aggregated value from column2, pivot table is obtained from the training set
#For example, it can be used to replace the empty cells in Age, using the mean Age of each title (Ms, Mr, Miss, etc)
def fill_colname1_by_train_colname2(df,train, colname1, colname2):
    bool1 = pd.isnull(df[colname1])
    if sum(bool1) > 0:
        print('\nimputing {0} using aggregated info baesd on name_title_adv from the training set.......\n'.format(colname1))
        colname1_by_colname2_table_train = pd.pivot_table(train, index = colname2, values = colname1)

        nan_colname1_by_colname2_table = pd.pivot_table(df[bool1], index = colname2, 
                                    values = 'Ticket_firstletter', aggfunc = 'count')
        for index in nan_colname1_by_colname2_table.index:
            value_from_pivot_table = colname1_by_colname2_table_train.loc[index][colname1]
            bool1 = pd.isnull(df[colname1]) 
            bool2 = df[colname2] == index
            to_replace = np.logical_and(bool1, bool2)
            df.loc[to_replace,colname1] = value_from_pivot_table
            print('filled {0} title {1} with age value {2}'.format(sum(to_replace),index, value_from_pivot_table))
    else:
        print('\nNo NaN in {0}\n'.format(colname1))
    return df

#fill in the empty cells using the training set mode value
def fill_with_train_mode(df,colname, train):
    bool1 = pd.isnull(df[colname])
    if sum(bool1) > 0:
        print('\nimputing {0} using mode from the training set......\n'.format(colname))
        df[colname] = df[colname].fillna(train[colname].mode()[0])
        print('{0} {1} imputed with {2}\n'.format(sum(bool1), colname, train[colname].mode()[0]))
    else:
        print('\nNo NaN in {0}\n'.format(colname))
    return df

#fill in the empty cells using the training set median value
def fill_with_train_median(df, colname, train):
    bool1 = pd.isnull(df[colname])
    if sum(bool1) > 0:
        print('\nimputing {0} using median from the training set......\n'.format(colname))
        df[colname] = df[colname].fillna(train[colname].median())
        print('{0} {1} imputed\n'.format(sum(bool1), colname))
    else:
        print('No NaN in {0}\n'.format(colname))
    return df

#convert a numeric variable to categorical
def convert_col_to_str(df, colname):
    df[colname] = df[colname].astype(str)
    return df

#normalize a variable
def normalize(df, colname):
    df['norm_' + colname] = np.log(df[colname] + 1)
    return df

#scale the numeric columns
from sklearn.preprocessing import StandardScaler
def apply_scaler(train, test, target):
    scale = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    colnames = train_scaled.select_dtypes(include=['float64','int64']).columns.to_list()
    colnames.remove(target)
    print('\nThe following columns are scaled:\n')
    print(colnames)
    scale.fit(train_scaled[colnames])
    train_scaled[colnames] =  scale.transform(train_scaled[colnames])
    test_scaled[colnames] =  scale.transform(test_scaled[colnames])
    return train_scaled, test_scaled

#if value_counts < 20 or if the variable is not numeric, print the value_count table
#if the variable is numeric, make density plot for frequency, and lineplot for interaction with the target variable
#if the variable is categorical and number of category is less than 20, 
#make countplot for counts, and boxplots for interaction with the target variable
def feature_plot(df, target, col): 
    categories = df[col].value_counts().index.to_list()
    n_categories = len(categories)
    if n_categories < 20 or df[col].dtype != 'int64':
        print(df[col].value_counts())
    
    plt.figure(figsize = (14,10))
    if df[col].dtype == 'int64':
        plt.subplot(2,1,1)
        if n_categories < 30:
            sns.distplot(df[col], bins = n_categories)
        else:
            sns.distplot(df[col], bins = 20)           
        plt.subplot(2,1,2)
        sns.lineplot(x = col, y = target, data = df)
    else:
        if n_categories < 20:
            plt.subplot(2,1,1)
            sns.countplot(x = col, data = df) 
            plt.subplot(2,1,2)
            sns.boxplot(x = col, y = target, data = df)
    plt.show()

def study_outliers(df, col):
    stat = df[col].describe()
    IQR = stat['75%'] - stat['25%']
    upper = stat['75%'] + 1.5 * IQR
    lower = stat['25%'] - 1.5 * IQR
    print('The upper and lower bounds for variable {} are {} and {}'.format(col, upper, lower))
    return lower, upper

#transform a variable based on the average of the target variable of each value
#for example, transfor each industry into the averaged salary of each industry
def transform_categorical(df, col, target, training_df): 
    category_mean = {}
    value_list = df[col].cat.categories.tolist()
    for value in value_list:
        category_mean[value] = training_df[training_df[col] == value][target].mean()
    df[col+'_transformed'] = df[col].map(category_mean)
    df[col+'_transformed'] = df[col+'_transformed'].astype('int64')
    return df

def convert_to_category(df, col):
    df[col] = df[col].astype('category')
    return df

def drop_duplicates(df, col):
    df = df.drop_duplicates(subset = col)
    return df

#get the variable list that will need convert to category and then encode
def salary_get_variable_list():
    variable_list = ['companyId', 'jobType', 'degree', 'major', 'industry']
    return variable_list

#make sure training and test has the same categorical variables
def encode_categorical(training, test):
    from sklearn import preprocessing
    cols = training.select_dtypes(include=['category']).columns.to_list()
    for col in cols:
        le = preprocessing.LabelEncoder()
        le.fit(training[col])
        training[col+'_encoded'] = le.transform(training[col])
        test[col+'_encoded'] = le.transform(test[col])
    return training, test

def salary_preprocess():
    #define constants
    variable_list = salary_get_variable_list()
    target = 'salary'
    
    #read the data and merge feature and salary for the training data
    features = pd.read_csv('data/train_features.csv')
    salaries = pd.read_csv('data/train_salaries.csv')
    test = pd.read_csv('data/test_features.csv')
    training = pd.merge(features,salaries, how = 'inner', on = 'jobId')

    #remove duplicates
    training = drop_duplicates(training,'jobId')
    test = drop_duplicates(test,'jobId')

    #remove salary = 0 in the training set
    training = training.drop(training[training[target]==0].index)

    #convert object to categorial variables
    #and transform them based on mean target (salary)
    for variable in variable_list:
        training = convert_to_category(training, variable)
        training = transform_categorical(training, variable, target, training)
        test = convert_to_category(test, variable)
        test = transform_categorical(test, variable, target, training)

    #encode categorical variables to dummies
    training, test = encode_categorical(training, test)
    
    #save scaler for later
    
    #print results on the screen
    training.info()
    training.head()
    test.info()
    test.head()
    return training, test

#Cross validates the model for 20% (cv=5) of the training set
from sklearn.model_selection import cross_val_score
def cross_val_model(model, feature_df, target, n_procs, mean_mse, cv_std):
    neg_mse = cross_val_score(model, feature_df, target, cv = 5, n_jobs = n_procs, 
                              scoring = 'neg_mean_squared_error')
    mean_mse[model] = -1.0 * np.mean(neg_mse)
    cv_std[model] = np.std(neg_mse)
    
#print a short summary
def print_summary(model, mean_mse, cv_std):
    print('\nmodel:\n', model)
    print('Average MSE:\n', mean_mse[model])
    print('Standard deviation during cross validation:\n', cv_std[model])

#feature importance
def get_model_feature_importances(model, feature_df):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = [0] * len(feature_df.columns)
    
    feature_importances = pd.DataFrame({'feature': feature_df.columns, 'importance': importances})
    feature_importances.sort_values(by = 'importance', ascending = False, inplace = True)
    ''' set the index to 'feature' '''
    feature_importances.set_index('feature', inplace = True, drop = True)
    return feature_importances
