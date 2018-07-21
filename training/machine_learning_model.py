import numpy as np
import pandas as pd
#View all columns
pd.set_option('display.max_columns', 200)

#sklearn imports
from sklearn.preprocessing import LabelEncoder

#fs management
import os
import matplotlib.pyplot as plt
import seaborn as sns


#listing directory, viewing files available
print(os.listdir("./Dataset"))

apptrain = pd.read_csv('./Dataset/application_train.csv')
apptest = pd.read_csv('./Dataset/application_test.csv')
apptrain.head()

print(apptest.shape)
apptest


# missing value check
def check_missing_value(df):
    # Returns how many values are missing in each column
    missing_values = df.isnull().sum()
    # Percentage
    missing_values_percent = 100 * missing_values / len(df)
    # Table
    missing_values_table = pd.concat([missing_values, missing_values_percent], axis=1)
    # renaming columns
    missing_value_table_renamed_columns = missing_values_table.rename( columns={0: 'Missing values', 1: '% of total values'})
    # return summary info
    return missing_value_table_renamed_columns


# Using previously defined function to create a new df to inspect columns
expl_missing_values_df = check_missing_value(apptrain)


mvdf = expl_missing_values_df.loc[~(expl_missing_values_df==0).all(axis=1)]
mvdf.sort_values(by=['% of total values'], ascending=False).head(30)

apptrain['TARGET'].value_counts()


fig, ax = plt.subplots()
x = np.arange(2)
plt.bar(x, [apptrain['TARGET'].value_counts()[0], apptrain['TARGET'].value_counts()[1]])
plt.xticks(x, ('High ability', 'Low ability'))
plt.show()

apptrain.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_count = 0

# Iterate through columns
for col in apptrain:
    if apptrain[col].dtype == "object":
        if len(list(apptrain[col].unique())) <= 2:
            #train on the training data
            le.fit(apptrain[col])
            #transform both training and testing data
            apptrain[col] = le.transform(apptrain[col])
            apptest[col] = le.transform(apptest[col])

            le_count += 1


print('{} columns were label encoded'.format(le_count))


#One-Hot encoding
apptrain = pd.get_dummies(apptrain)
apptest = pd.get_dummies(apptest)

print('Training features shape: {}'.format(apptrain.shape))
print('Training features shape: {}'.format(apptest.shape))


train_labels = apptrain['TARGET']

#aligning the training and testing data, keep only columns present in both df's
apptrain, apptest = apptrain.align(apptest, join = 'inner', axis = 1)
apptrain['TARGET'] = train_labels

print('Training Features shape: ', apptrain.shape)
print('Testing features shape: ', apptest.shape)


(apptrain['DAYS_BIRTH'] / -365).describe()
apptrain['DAYS_EMPLOYED'].describe()



# Seems as if our data has an error, Some people have been employed for 100+ years,
fig1, ax1 = plt.subplots()
x = np.arange(2)
plt.bar(x, [apptrain[apptrain['DAYS_EMPLOYED'] == 365243].count()[0], apptrain[apptrain['DAYS_EMPLOYED'] != 365243].count()[0] ])
plt.xticks(x, ['Outlying', 'Non outlying'])
plt.show()





de_anomalous = apptrain[apptrain['DAYS_EMPLOYED'] == 365243]
de_non_anomalous = apptrain[apptrain['DAYS_EMPLOYED'] != 365243]

print('Non-anomalous dataset defaults on {}%'.format(100*de_non_anomalous['TARGET'].mean()))
print('Anomalous dataset defaults on {}%'.format(100*de_anomalous['TARGET'].mean()))



apptrain['DAYS_EMPLOYED_ANOM'] = apptrain['DAYS_EMPLOYED'] == 365243
apptrain['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
apptrain['DAYS_EMPLOYED'].plot.hist(title="Days employment Histogram")
plt.xlabel('Days Employment')



# Very important to migrate changes over to our test data as well.
apptest['DAYS_EMPLOYED_ANOM'] = apptest['DAYS_EMPLOYED'] == 365243
apptest['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
print('There are {} anomalies in the test data out of {} entries'.format(apptest['DAYS_EMPLOYED_ANOM'].sum(), len(apptest)))


correlations = apptrain.corr()['TARGET'].sort_values()
print('Top 15 positive correlations', correlations.tail(15))
print('Top 15 negative correlations', correlations.head(15))



apptrain['DAYS_BIRTH'] = abs(apptrain['DAYS_BIRTH'])
apptrain['DAYS_BIRTH'].corr(apptrain['TARGET'])


plt.style.use('fivethirtyeight')
plt.hist(apptrain['DAYS_BIRTH']/365, bins=50, edgecolor='y')
plt.title('Age of Client')
plt.xlabel('Age(years)')
plt.ylabel('Count')



plt.figure(figsize = (10, 8))
sns.kdeplot(apptrain.loc[apptrain['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'TARGET == 0')
sns.kdeplot(apptrain.loc[apptrain['TARGET'] == 1, 'DAYS_BIRTH'] / 365, LABEL = 'TARGET == 1')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Distribution of ages')



age_df = apptrain[['TARGET', 'DAYS_BIRTH']]
age_df['YEARS_BIRTH'] = age_df['DAYS_BIRTH'] / 365

age_df['YEARS_BINNED'] = pd.cut(age_df['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_df.head(15)




age_groups = age_df.groupby('YEARS_BINNED').mean()
age_groups


plt.figure(figsize = ( 20, 8))

plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
plt.title('Failure to repay by binned age')




external_data = apptrain[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
external_data_correlation = external_data.corr()
external_data_correlation


plt.figure(figsize = (8, 6))
sns.heatmap(external_data_correlation, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')


plt.figure(figsize = (15,12))

for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    # Create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    sns.kdeplot(apptrain.loc[apptrain['TARGET'] == 0, source], label = 'TARGET == 0')
    sns.kdeplot(apptrain.loc[apptrain['TARGET'] == 1, source], label = 'TARGET == 1')

    plt.title('Distribution of {} by target value'.format(source))
    plt.xlabel('{}'.format(source))
    plt.ylabel('Density')

plt.tight_layout(h_pad = 2.5)


#Creating polynomial features
poly_features = apptrain[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = apptest[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')

poly_target = poly_features['TARGET']
poly_features = poly_features.drop(columns = ['TARGET'])

#Imputing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures
#Create polynomial object with specific degree
poly_transformer = PolynomialFeatures(degree = 3)


#train the polynomial features
poly_transformer.fit(poly_features)

#transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial features shape: ', poly_features.shape)



poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]


# Create a dataframe with the newly created features
poly_features = pd.DataFrame(poly_features, columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))

# Add back the target
poly_features['TARGET'] = poly_target

#run correlations for new features
poly_corrs = poly_features.corr()['TARGET'].sort_values()

# Display most negative and most positive
print(poly_corrs.head(10))
print(poly_corrs.tail(5))



#Create a dataframe with newly created features for test
poly_features_test = pd.DataFrame(poly_features_test, columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))


#Putting features into training dataframe
poly_features['SK_ID_CURR'] = apptrain['SK_ID_CURR']
apptrain_poly = apptrain.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

#Merge polynomial features into testing dataframe
poly_features_test['SK_ID_CURR'] = apptest['SK_ID_CURR']
apptest_poly = apptest.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

#align DFs
apptrain_poly, apptest_poly = apptrain_poly.align(apptest_poly, join = 'inner', axis = 1)
print('training data with polynomial features shape: ', apptrain_poly.shape)
print('testing data with polynomial features shape: ', apptest_poly.shape)




apptrain_domain = apptrain.copy()
apptest_domain = apptest.copy()

apptrain_domain['CREDIT_INCOME_PERCENT'] = apptrain_domain['AMT_CREDIT'] * 100 / apptrain_domain['AMT_INCOME_TOTAL']
apptrain_domain['ANNUITY_INCOME_PERCENT'] = apptrain_domain['AMT_ANNUITY'] / apptrain_domain['AMT_INCOME_TOTAL']
apptrain_domain['CREDIT_TERM'] = apptrain_domain['AMT_ANNUITY'] / apptrain_domain['AMT_CREDIT']
apptrain_domain['DAYS_EMPLOYED_PERCENT'] = apptrain_domain['DAYS_EMPLOYED'] / apptrain_domain['DAYS_BIRTH']


apptest_domain['CREDIT_INCOME_PERCENT'] = apptest_domain['AMT_CREDIT'] / apptest_domain['AMT_INCOME_TOTAL']
apptest_domain['ANNUITY_INCOME_PERCENT'] = apptest_domain['AMT_ANNUITY'] / apptest_domain['AMT_INCOME_TOTAL']
apptest_domain['CREDIT_TERM'] = apptest_domain['AMT_ANNUITY'] / apptest_domain['AMT_CREDIT']
apptest_domain['DAYS_EMPLOYED_PERCENT'] = apptest_domain['DAYS_EMPLOYED'] / apptest_domain['DAYS_BIRTH']


from sklearn.preprocessing import MinMaxScaler, Imputer

if 'TARGET' in apptrain:
    train = apptrain.drop(columns = ['TARGET'])
else:
    train = apptrain.copy()


features = list(train.columns)

#copy test data
test = apptest.copy()

#median strategy imputing
imputer = Imputer(strategy='median')

#scale each feature
scaler = MinMaxScaler(feature_range=(0, 1))

#fitting on training data
imputer.fit(train)

#transforming training and testing data
train = imputer.transform(train)
test = imputer.transform(apptest)

#Fit scales
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data sape: ', train.shape)
print('testing data shape: ', test.shape)



#Logistic regression model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=0.0001)

#train on training dataset
log_reg.fit(train, train_labels)

log_reg_pred = log_reg.predict_proba(test)[:, 1]

submit = apptest[['SK_ID_CURR']]
submit['TARGET'] = log_reg_preds
