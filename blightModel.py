# Script to make the ML function

import pandas as pd
import numpy as np

# Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier

def blight_model():
    
    # init data frames :)
    dfTrain = pd.read_csv('dataframes/train.csv', engine='python')
    dfTest = pd.read_csv('dataframes/test.csv', engine='python')
    dfAddresses = pd.read_csv('dataframes/addresses.csv', engine='python')
    dfGeo = pd.read_csv('dataframes/latlons.csv', engine='python')
    
    # Clean data
    dfTrain2 = dfTrain.copy(deep=True)
    # Drop rows with target column with NAN
    dfTrain2 = dfTrain2[pd.notnull(dfTrain2['compliance'])]
    # Print number of NAs per column
    # dfTrain2.isnull().sum()

    # Drop column with full NAN or with almost all NAN
    dfTrain2 = dfTrain2.drop(['violation_zip_code', 'grafitti_status', 'non_us_str_code'], axis=1)
    
    # Drop leak data (google "Data Science Leak Data" for more information)
    dfTrain2 = dfTrain2.drop(['compliance_detail', 'collection_status', 'payment_amount', 
                          'payment_date', 'payment_status', 'balance_due'], axis=1)
    
    # Add Geo addresses from the GeoDataFrame to the main data frame
    newdf = dfAddresses.merge(dfGeo, on=['address'])
    dfTrain2 = dfTrain2.merge(newdf, on=['ticket_id'])

    # Initialize df for testing from the clean data frame
    dfTest2 = dfTest.merge(newdf, on=['ticket_id'])
    
    #We explore the remainning features
    #for i in range(len(dfTrain2.columns)):
    # print(dfTrain2.groupby(dfTrain2.columns[i]).size())
    # print('')
    
    # Far below there are plots and grouppedBy that I used to have a deeper understanding
    # a) admin_fee and state_fee are constant in 100% of all the rows, country is mainly USA, cleanup cost is 0
    # b) Late_fee, its a fee that is placed if its not paid on time
    # c) Hearing date as well as late_fee can induce to data leak
    # d) judgment_amount includes all fees, it diverges from fine_fee, and it can include data leak from late_fee
    # e) ticket_ID the ID of each row is not meaningfull, ticket_issued_date doesnt impact in the violator
    # f) inspector_name is a not a variable that impacts in the violator, the violator is independant from it
    # g) agency_name same as inspector_name
    # h) violator_name appears several times with different instances with misspells. Name shouldnt be relevant, we drop it
    # i) Each violator_name becomes a new instance category, only relevant if the same violator name (the same category)
    #    is already in the system. Therefore if Max is in the system 10 times and she hasnt paid anything, on the new 11th instance
    #    the prediction is that she won't pay. This is useless.
    dfTrain2 = dfTrain2.drop(['admin_fee', 'state_fee', 'country', 'ticket_issued_date',
                              'hearing_date',
                              #'late_fee',
                              'ticket_id',
                              'clean_up_cost', 'judgment_amount', 'inspector_name', 'violator_name',
                              'agency_name'], axis=1)
    
    # We continue to explore the features with histograms
    dfTrainP = dfTrain2.loc[dfTrain2['compliance'] == 1.0].copy()
    dfTrainN = dfTrain2.loc[dfTrain2['compliance'] == 0.0].copy()
    
    # Init an array with string to work as a filter
    features = ['zip_code', 'violation_code', 'fine_amount', 'discount_amount', 
                'lat', 'lon', 'disposition', 'late_fee']
    
    # Obtain the features from the clean data frame train in a new copy
    X_traina = dfTrain2.filter(features).copy(deep = True)
    #Obtain label data
    y = dfTrain2.filter(['compliance']).copy(deep = True)
    
    # Obtain features for dfTest2
    dfTest2 = dfTest2.filter(features)
    
    # Perform deep copy to have a traceability
    X_train2 = X_traina.copy(deep=True)
    X_testSubmit = dfTest2.copy(deep=True)
    
    # Obtain the type of each column (feature)
    Xdata = X_traina.dtypes

    # Change the value of the 'Object' feautres to a categorial value
    for i in range(len(Xdata)):
        if Xdata[i] =='O':
            column = Xdata.index[i]
            #Converts the strings and object types into int64 or float       
            X_train2[column] = LabelEncoder().fit_transform(X_train2[column].astype(str))
            X_testSubmit[column] = LabelEncoder().fit_transform(X_testSubmit[column].astype(str))
            # This makes the column of unique categorial value, meaning that number 3 is not bigger than 2, 
            # its just an instance named 3
            pd.Categorical(X_train2[column],categories=X_train2[column].unique())
            pd.Categorical(X_testSubmit[column],categories=X_testSubmit[column].unique())
    
    #Input mean values to NANs
    my_imputer = Imputer()
    X_train2 = my_imputer.fit_transform(X_train2)
    X_testSubmit = my_imputer.fit_transform(X_testSubmit)
    
    # Get the scalar
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(X_train2, y, random_state = 0)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use ML algorithms

    # Knn (not the best approach for this data set)
    # knn = KNeighborsClassifier(n_neighbors = 75).fit(X_train_scaled, y_train)
    
    # Tree clasiffier
    # clf = DecisionTreeClassifier(random_state=0).fit(X_train_scaled, y_train)
    # clf = RandomForestClassifier(random_state=0).fit(X_train_scaled, y_train)

    # Tree gradient boost is the one that performed best
    clf = GradientBoostingClassifier(random_state=0, n_estimators=50, learning_rate=0.5).fit(X_train_scaled, y_train)
    
    y_scores = clf.predict_proba(X_test_scaled)
    fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
    roc_auc = auc(fpr, tpr)

    # For this set we obtained: 0.814747374362
    print("The AUC for this set is: ",roc_auc)
    
    #Now we predict the Test to submit set
    X_test_scaled_submit = scaler.transform(X_testSubmit)
    y_scores_Submit = clf.predict_proba(X_test_scaled_submit)
    
    # We make a new series of two columns, first column is the ticke id, second column is the prediction to be paid on time
    # The name of the series is "compliance"
    dfaux = pd.DataFrame(y_scores_Submit)
    dfaux['ticket_id'] = dfTest['ticket_id']
    s = pd.Series(dfaux[1].values, name = 'compliance',
                  index=pd.Index(dfaux['ticket_id'], name='ticket_id'))
    
    return s