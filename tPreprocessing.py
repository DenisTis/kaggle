import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

def adjustDataframe(dataFrame):
##  Get name prefix
    namePrefix = dataFrame["Name"].str.split(', ', 1).str[1]
    namePrefix = namePrefix.str.split('. ',1).str[0]
    dataFrame["NamePrefix"] = namePrefix
##  Split ticket into numeric and prefix parts
    spacedTicket = ' ' + dataFrame['Ticket'].astype(str)
    ticketSplit = spacedTicket.str.rsplit(' ',1,expand=True)
    dataFrame["TicketPrefix"] = ticketSplit[0]
    dataFrame["TicketPrefix"] = dataFrame["TicketPrefix"].str.replace(".","")
    dataFrame["TicketPrefix"] = dataFrame["TicketPrefix"].str.replace(" ","")
    dataFrame["TicketNumber"] = ticketSplit[1]  
#Check if there were NaN for those columns
    dataFrame["FamilySize"] = dataFrame["SibSp"] + dataFrame["Parch"]  + 1

## There are multiple cabins per some of passengers - how to proceed?
# For now I will only use deck name and amount of cabins. For later I should check how could I use multiple entries
    dataFrame["CabinDeck"] = dataFrame["Cabin"].str[:1]
    dataFrame["CabinsAmount"] = dataFrame["Cabin"].str.split(" ",-1).str.len()

    del namePrefix, spacedTicket, ticketSplit
    dataFrame.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
    return dataFrame

def getValidColumns(columns):
    allNumericColumns = ["Pclass", "Age", "FamilySize", "Fare", "CabinsAmount"]
    allCategoricalColumns = ["Sex","NamePrefix", "TicketPrefix", "CabinDeck", "Embarked"]
    numColumns = []
    catColumns = []

    for column in columns:
        if column in allNumericColumns:
            numColumns.append(column)
        if column in allCategoricalColumns:
            catColumns.append(column)
    return numColumns, catColumns

def preprocess_dataset(columns, dropColumn=None, degree = 1):
    yColumn = "Survived"
    dfTrain = pd.read_csv("./train.csv")
    dfTest = pd.read_csv("./test.csv")

    y = dfTrain[yColumn]
    dfTrain.drop(yColumn, axis=1, inplace=True)
    X_full = dfTrain.append(dfTest, ignore_index=True)
    #   Enhance dataframe with additional columns
    X_full = adjustDataframe(X_full)
    # Fill nan values
    X_full = X_full.replace(r'^\s*$', np.nan, regex=True)
    # Impute nan to mean and encode categorical data into separate columns
    numericColumns, categoricalColumns = getValidColumns(columns)
    mean_Imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_numeric = mean_Imputer.fit_transform(X_full[numericColumns])
    X_numeric = StandardScaler().fit_transform(X_numeric)

    polyFeatures = PolynomialFeatures(degree)
    polyFeatures.fit_transform(X_numeric)

    categorical_data = X_full[categoricalColumns].fillna("N/A").astype(str)
    oneHotEncoder = OneHotEncoder(sparse = False,drop=dropColumn)
    X_categorical = oneHotEncoder.fit_transform(categorical_data)
    
    # columnNames = numericColumns
    columnNames = list(polyFeatures.get_feature_names())
    columnNames.extend(list(oneHotEncoder.get_feature_names()))

    # Split data back to train and test
    X = np.hstack((X_numeric,X_categorical))
    X_train = X[0:891,:]
    X_final_test = X[891:1309,:]
    # Remove columns with low correlation
    delColumns, columnNames = removeLowCorrelatedColumns(X_train, y, columnNames, 0.05)
    X_train = np.delete(X_train, delColumns, axis = 1)
    X_final_test = np.delete(X_final_test, delColumns, axis = 1)
    return X_train, X_final_test, y, columnNames


def removeLowCorrelatedColumns(dataset,y, columnNames, coef):
    columnsAmount = dataset.shape[1]
    LCColumns = []
    LCColumnNames = []
    for column in range(columnsAmount):
        corr = np.corrcoef(dataset[:, column], y) [1,0]
        if corr < coef and corr > (-1 * coef):
            LCColumns.append(column)
            LCColumnNames.append(columnNames[column])
        if corr == np.nan:
            LCColumns.append(column)
            LCColumnNames.append(columnNames[column])
        # print(columnNames[column], ", coef: ",round(np.corrcoef(dataset[:, column], y) [1,0],3))
    # dataset = np.delete(dataset, LCColumns,axis=1)
    print("Deleted columns: ", LCColumnNames)
    leftColumnNames = [x for x in columnNames if x not in LCColumnNames]
    return LCColumns, leftColumnNames
#--------------------------------
# columnsAmount = X_train.shape[1]
# for column in range(columnsAmount):
#     corr = np.corrcoef(X_train[:, column], y) [1,0]
#     print(columnNames[column], ", coef: ",round(np.corrcoef(X_train[:, column], y) [1,0],3))


# used_columns= ["Pclass", "Age", "FamilySize", "Fare", "CabinsAmount","Sex", "NamePrefix"]
# X_train, X_test, y, columnNames = preprocess_dataset(used_columns, dropColumn="first")
# columnsAmount = X_train.shape[1]

# LCColumns = []
# LCColumnNames = []
# for column in range(columnsAmount):
#     corr = np.corrcoef(X_train[:, column], y) [1,0]
#     if corr < 0.05 and corr > -0.05:
#         LCColumns.append(column)
#         LCColumnNames.append(columnNames[column])
#     print(columnNames[column], ", coef: ",round(np.corrcoef(X_train[:, column], y) [1,0],3))
# X_train = np.delete(X_train, LCColumns,axis=1)
# print(LCColumnNames)
# columnNames = [x for x in columnNames if x not in LCColumnNames]
# print(columnNames)

# columnsAmount = X_train.shape[1]
# for column in range(columnsAmount):
#     corr = np.corrcoef(X_train[:, column], y) [1,0]
#     print(columnNames[column], ", coef: ",round(np.corrcoef(X_train[:, column], y) [1,0],3))
