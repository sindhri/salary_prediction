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
def apply_scaler(train, test):
    scale = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    colnames = list(train_scaled.select_dtypes(include=['float64','int64']).columns)
    print('\nThe following columns are scaled:\n')
    print(colnames)
    train_scaled[colnames] =  scale.fit_transform(train_scaled[colnames])
    test_scaled[colnames] =  scale.transform(test_scaled[colnames])
    return [train_scaled, test_scaled]

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
    return [lower, upper]

def label_encode(df, col, target): 
    ''' Encodes the categories of the column based on the mean value of the salary of every category
    in order to replace the label of the category '''
    cat_mean = {}
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_mean[cat] = df[df[col] == cat][target].mean()
    df[col] = df[col].map(cat_mean)
    return df[col]

def convert_to_category(df, col):
    df[col] = df[col].astype('category')
    return df

def drop_duplicates(df, col):
    df = df.drop_duplicates(subset = col)
    return df

def encode_category(df, target):
    df_copy = df[:]
    for col in df_copy.columns:
        if df_copy[col].dtype.name == 'category':
            df_copy[col]=label_encode(df_copy, col, target)
            df_copy[col] = df_copy[col].astype('int64')
    return df_copy

#chain the whole preprocess for the training data
def preprocess_training(df):
    df = drop_duplicates(df,'jobId')
    
    #convert to categorial variables
    df = convert_to_category(df, 'companyId')    
    df = convert_to_category(df, 'jobType')
    df = convert_to_category(df, 'degree')
    df = convert_to_category(df, 'major')
    df = convert_to_category(df, 'industry')
    
    #remove salary = 0
    df = df.drop(training[df['salary']==0].index)
    
    #encode categorical variables
    df_copy = encode_category(df, 'salary)

    return df_copy
                              
def preprocess_test(df):
    df = drop_duplicates(df,'jobId')

    #convert to categorial variables
    df = convert_to_category(df, 'companyId')    
    df = convert_to_category(df, 'jobType')
    df = convert_to_category(df, 'degree')
    df = convert_to_category(df, 'major')
    df = convert_to_category(df, 'industry')
        
    #encode categorical variables
    df_copy = encode_category(df, 'salary)

    return df_copy
