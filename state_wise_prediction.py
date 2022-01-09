import pandas as pd, numpy as np, os, seaborn as sns
from matplotlib import pyplot as plt
from dask import dataframe as dd
from sklearn.utils import shuffle
from lazypredict.Supervised import LazyClassifier, LazyRegressor
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC 
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
%matplotlib qt



df = pd.read_csv('SHR76_19.csv', delimiter=',')

df = df.sort_values(by=['State'], ascending=True ) # sort to reflect correct state-wise alphabetical order


''' data processing '''

### homicide overlap variable
df['Year2']=df['Year'].copy()
df['Year2']=df['Year2'].astype(str)
df['Year2'] = str('Year2')
df['Code_Overlap'] = df['State']+"_"+df['Agency']+"_"+df['Year2']+"_"+df['Month']
df['Code_Overlap+Agency'] = df['State']+"_"+df['Agency']+"_"+df['Agentype']+"_"+df['Year2']+"_"+df['Month']
df['Monthly_Overlap']= df.duplicated('Code_Overlap')
df['Monthly_Agency_Overlap']=df.duplicated('Code_Overlap+Agency')


df['VicCount'] = df['VicCount']+1
df['OffCount'] = df['OffCount']+1

df_red = df[['ID', 'State', 'Year', 'Month', 'Solved', 'Homicide', 'VicAge', 'VicSex', 
             'Circumstance','Weapon', 'Agentype', 'VicRace', 'VicCount', 'OffCount', 
             'Monthly_Overlap']]


### decade function
def decade(Year):
    if Year > 1970 and Year < 1980:
        decade='70s'
        
    elif Year >= 1980 and Year < 1990:
        decade='80s'
        
    elif Year >= 1990 and Year < 2000:
        decade='90s'
        
    elif Year >=2000 and Year <2010:
        decade='00s'

    else:
        decade='10s'

    return decade

df_red['Decade'] = df.apply(lambda x: decade(x['Year']), axis=1) # apply the created function


# age function

df_red = df_red[df_red.VicAge !=999] # remove unknown age

def age_cat(VicAge):
    if VicAge >=0 and VicAge <=5:
        age_cat='0-5'
    elif VicAge >5 and VicAge <=10:
        age_cat='6-10'
    elif VicAge >10 and VicAge <=15:
        age_cat='11-15'
    elif VicAge >15 and VicAge <=20:
        age_cat='16-20'
    elif VicAge >20 and VicAge <=25:
        age_cat='21-25'
    elif VicAge >25 and VicAge <=30:
        age_cat='26-30'
    elif VicAge >30 and VicAge <=35:
        age_cat='31-35'
    elif VicAge >35 and VicAge <=40:
        age_cat='36-40'
    elif VicAge >40 and VicAge <=45:
        age_cat='41-45'
    elif VicAge >45 and VicAge <=50:
        age_cat='46-50'
    elif VicAge >50 and VicAge <=55:
        age_cat='51-55'
    elif VicAge >55 and VicAge <=60:
        age_cat='56-60'
    elif VicAge >60 and VicAge <=65:
        age_cat='61-65'
    elif VicAge >65 and VicAge <=70:
        age_cat='66-70'
    elif VicAge >70 and VicAge <=75:
        age_cat='71-75'
    elif VicAge >75 and VicAge <=80:
        age_cat='76-80'
    elif VicAge >80 and VicAge <=85:
        age_cat='81-85'
    elif VicAge >85 and VicAge <=90:
        age_cat='86-90'
    elif VicAge >90 and VicAge <=95:
        age_cat='91-95'
    elif VicAge >95 and VicAge <=99:
        age_cat='96-99'
    else:
        age_cat='999'
    return age_cat

df_red['Age_Cat'] = df_red.apply(lambda x: age_cat(x['VicAge']), axis=1) 



#create a data frame dictionary to store your data frames
UniqueStates = df.State.unique()
DataFrameDict = {elem : pd.DataFrame for elem in UniqueStates}


for key in DataFrameDict.keys():
    DataFrameDict[key] = df_red[:][df_red.State == key]
    
for key in DataFrameDict.keys():
    DataFrameDict[key] =  pd.get_dummies(DataFrameDict[key], columns = ['Decade', 'Month', 'Homicide', 
                                                                        'VicSex', 'Age_Cat', 'Weapon', 'Circumstance', 
                                                                        'Agentype', 'VicRace',  'Monthly_Overlap','Solved'], drop_first=False)
#drop useless variables, and shuffle 
for key in DataFrameDict.keys():
    DataFrameDict[key].drop(['ID','Year','Homicide_Murder and non-negligent manslaughter', 
                             'VicAge', 'State','Monthly_Overlap_False','Solved_No'], axis=1, inplace=True)
    DataFrameDict[key]=shuffle(DataFrameDict[key])


#rename for better visualization
for key in DataFrameDict.keys():
    DataFrameDict[key].rename(columns={"VicCount":"N of Victims", "OffCount": "N of Offenders", "Decade_10s": "Decade: 2010s", "Decade_70s": "Decade: 1970s", "Decade_80s": "Decade: 1980s",
                             "Decade_90s": "Decade: 1990s", "Month_August": "Month: August", "Month_December": "Month: December", "Month_February": "Month: February",
                             "Month_January": "Month: January", "Month_July": "Month: July", "Month_June": "Month: June", "Month_March": "Month: March", "Month_May": "Month: May",
                             "Month_November": "Month: November", "Month_October": "Month: October", "Month_September": "Month: September",
                             "Homicide_Murder and non-negligent manslaughter": "Type: Murder and Non-negl. Manslaughter", "VicSex_Female": "Victim Sex: Female","VicSex_Male": "Victim Sex: Male",
                             "VicSex_Unknown": "Victim Sex: Unknown", "Age_Cat_0-5":"Age: 0-5", "Age_Cat_11-15": "Age: 11-15", "Age_Cat_16-20": "Age: 16-20", "Age_Cat_21-25": "Age: 21-25",
                             "Age_Cat_26-30": "Age: 26-30", "Age_Cat_31-35": "Age: 31-35", "Age_Cat_36-40": "Age: 36-40", "Age_Cat_41-45": "Age: 41-45",
                             "Age_Cat_46-50": "Age: 46-50", "Age_Cat_51-55": "Age: 51-55", "Age_Cat_56-60": "Age 56-60", "Age_Cat_6-10": "Age: 6-10",
                             "Age_Cat_61-65": "Age: 61-65", "Age_Cat_66-70": "Age: 66-70", "Age_Cat_71-75": "Age 71-75", "Age_Cat_76-80":"Age: 76-80",
                             "Age_Cat_81-85": "Age: 81-85", "Age_Cat_86-90": "Age: 86-90", "Age_Cat_91-95": "Age: 91-95", "Age_Cat_96-99": "Age: 96-99",
                             "Weapon_Blunt object - hammer, club, etc": "Weapon: Blunt Object", "Weapon_Drowning": "Weapon: Drowning",
                             "Weapon_Explosives": "Weapon: Explosives", "Weapon_Fire": "Weapon: Fire", "Weapon_Firearm, type not stated": "Weapon: Firearm",
                             "Weapon_Handgun - pistol, revolver, etc": "Weapon: Handgun",
                             "Weapon_Knife or cutting instrument": "Weapon: Knife or Cutting Inst.",
                             "Weapon_Narcotics or drugs, sleeping pills": "Weapon: Narcotics, Drugs", "Weapon_Other gun": "Weapon: Other Gun",
                             "Weapon_Other or type unknown": "Weapon: Other/Unknown",
                             "Weapon_Personal weapons, includes beating": "Weapon: Personal Weapon (e.g. Beating)",
                             "Weapon_Poison - does not include gas": "Weapon: Poison",
                             "Weapon_Pushed or thrown out window": "Weapon: Pushed/Thrown OoW", "Weapon_Rifle": "Weapon: Rifle", "Weapon_Shotgun": "Weapon: Shotgun",
                             "Weapon_Strangulation - hanging": "Weapon: Strangulation/Hanging",
                             "Circumstance_All other manslaughter by negligence": "Circumstance: Other Manslaughter by Negligence",
                             "Circumstance_All suspected felony type": "Circumstance: All Susp. Felony Type",
                             "Circumstance_Argument over money or property": "Circumstance: Argument over Money/Prop.", "Circumstance_Arson": "Circumstance: Arson",
                             "Circumstance_Brawl due to influence of alcohol": "Circumstance: Brawl (Alcohol)",
                             "Circumstance_Brawl due to influence of narcotics": "Circumstance: Brawl (Narcotics)",
                             "Circumstance_Burglary": "Circumstance: Burglary", "Circumstance_Child killed by babysitter": "Circumstance: Killed by Babysitter",
                             "Circumstance_Children playing with gun": "Circumstance: Children Playing w/Gun",
                             "Circumstance_Circumstances undetermined": "Circumstance: Undetermined",
                             "Circumstance_Felon killed by police": "Circumstance: Felon Killed by Police",
                             "Circumstance_Felon killed by private citizen": "Circumstance: Felon Killed by Priv. Cit.", "Circumstance_Gambling": "Circumstance: Gambling",
                             "Circumstance_Gangland killings": "Circumstance: Gang-Related",
                             "Circumstance_Gun-cleaning death - other than self": "Circumstance: Gun-Cleaning Death",
                             "Circumstance_Institutional killings": "Circumstance: Institutional Killing",
                             "Circumstance_Juvenile gang killings": "Circumstance: Juveline Gang-related", "Circumstance_Larceny": "Circumstance: Larceny",
                             "Circumstance_Lovers triangle": "Circumstance: Lovers Triangle", "Circumstance_Motor vehicle theft": "Circumstance: Motor Vehicle",
                             "Circumstance_Narcotic drug laws": "Circumstance: Narcotics Drug Laws", "Circumstance_Other": "Circumstance: Other",
                             "Circumstance_Other - not specified": "Circumstance: Non Specified", "Circumstance_Other arguments": "Circumstance: Other Arguments",
                             "Circumstance_Other negligent handling of gun": "Circumstance: Other Negligent Handling of Gun",
                             "Circumstance_Other sex offense": "Circumstance: Other Sex Offense",
                             "Circumstance_Prostitution and commercialized vice": "Circumstance: Prostitution/Commercialized Vice",
                             "Circumstance_Rape": "Circumstance: Rape", "Circumstance_Robbery": "Circumstance: Robbery",
                             "Circumstance_Sniper attack": "Circumstance: Sniper Attack",
                             "Circumstance_Victim shot in hunting accident": "Circumstance: Hunting Accident",
                             "Agentype_Municipal police": "Agency: Municipal Police", "Agentype_Primary state LE": "Agency: Primary State LE",
                             "Agentype_Regional police": "Agency: Regional Police", "Agentype_Sheriff": "Agency: Sheriff",
                             "Agentype_Special police": "Agency: Special Police", "Agentype_Tribal": "Agency: Tribal", "VicRace_Asian": "Victim Race: Asian",
                             "VicRace_Black": "Victim Race: Black", "VicRace_Native Hawaiian or Pacific Islander": "Victim Race: Native, Hawaiian, Pacific Islander",
                             "VicRace_Unknown": "Victim Race: Unknown", "VicRace_White": "Victim Race: White", "Monthly_Overlap_True": "Monthly State/Agency Overlap"
                             }, inplace=True)

 
#create X and Y dataframes in dictionaries
XDict = {elem : pd.DataFrame for elem in UniqueStates}
YDict = {elem : pd.DataFrame for elem in UniqueStates}
for key in DataFrameDict.keys():
    YDict[key]=DataFrameDict[key].iloc[:,-1]
    XDict[key]=DataFrameDict[key].iloc[:,0:-1]

    
    
    
# perform train and test split for each state dataset stored in the dictionaries

import sklearn as sk
from sklearn.model_selection import train_test_split
X_train = []#{}
X_test = []
y_train = []
y_test = []


for key in DataFrameDict.keys():
    a,b,c,d = train_test_split(XDict[key], YDict[key], test_size=0.3, random_state=0)
    
    X_train.append(a)
    X_test.append(b)
    y_train.append(c)
    y_test.append(d)


X_train_d = {}
X_test_d = {}
Y_train_d = {}
Y_test_d = {}

for index, value in enumerate(X_train):
    X_train_d[index] = value

for index, value in enumerate(y_train):
    Y_train_d[index] = value

for index, value in enumerate(X_test):
    X_test_d[index] = value

for index, value in enumerate(y_test):
    Y_test_d[index] = value

# save dictionaries for reproducility 
import pickle   
state_Xtrain_sets = open("stateSHAP_Xtrain_sets.pkl","wb")
pickle.dump(X_train_d,state_Xtrain_sets )
state_Xtrain_sets.close()


state_Xtest_sets = open("stateSHAP_Xtest_sets.pkl","wb")
pickle.dump(X_test_d,state_Xtest_sets )
state_Xtest_sets.close()

state_Ytrain_sets = open("stateSHAP_Ytrain_sets.pkl","wb")
pickle.dump(Y_train_d,state_Ytrain_sets )
state_Ytrain_sets.close()

state_Ytest_sets = open("stateSHAP_Ytest_sets.pkl","wb")
pickle.dump(Y_test_d,state_Ytest_sets )
state_Ytest_sets.close()


#upload them
X_train_d = pickle.load(open("stateSHAP_Xtrain_sets.pkl", "rb"))
X_test_d =  pickle.load(open("stateSHAP_Xtest_sets.pkl", "rb"))
Y_train_d = pickle.load(open("stateSHAP_Ytrain_sets.pkl", "rb"))
Y_test_d =  pickle.load(open("stateSHAP_Ytest_sets.pkl", "rb"))      
    
 


# perform grid search for different XGBoost hyperparams for each state dataset

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# parameter grid
param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.001, 0.01, 0.1, 0.3, 0.5]}

# evaluation metrics
scoring = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']

# cross-validation splits
cv = StratifiedKFold(n_splits=5)

# results data frame
df_grid_results = pd.DataFrame()

for key in X_train_d.keys():

    # run the grid search
    grid_search = GridSearchCV(
        estimator=XGBClassifier(objective='binary:logistic', random_state=42), 
        param_grid=param_grid, 
        scoring=scoring, 
        cv=cv, 
        n_jobs=5, 
        verbose=2, 
        refit='balanced_accuracy'
    )
    
    grid_search.fit(X_train_d[key], Y_train_d[key])
    
    # save the grid search results in the data frame
    df_temp = pd.DataFrame(grid_search.cv_results_)
    df_temp['dataset'] = key
    
    df_grid_results = df_grid_results.append(df_temp, ignore_index=True)


# process resulting dataset for analysis    
df_grid_results = df_grid_results.set_index(df_grid_results['params'].apply(lambda x: '_'.join(str(val) for val in x.values()))).rename_axis('kernel')
df_grid_results.reset_index(inplace=True)
df_grid_results['hyperparams']=df_grid_results['kernel']
df_grid_results['Avg. Precision and Balanced Accuracy']=(df_grid_results['mean_test_balanced_accuracy']+df_grid_results['mean_test_precision'])/2
idx=df_grid_results.groupby(by='dataset')['Avg. Precision and Balanced Accuracy'].idxmax()





#save your model and results
import joblib
joblib.dump(grid_search, 'state_level_GridSearch02012022.pkl')
df_grid_results.to_csv('state_level_grid_results.csv')
#load your model for further usage
grid_search = joblib.load("state_level_GridSearch02012022.pkl")
df_grid_results = pd.read_csv("")



# derive best performing configuration for model fit via df_grid_results

idx=df_grid_results.loc[df_grid_results.groupby(['dataset'])['Avg. Precision and Balanced Accuracy'].idxmax()]
idx['hyperparams'].nunique() # list of unique best configurations, raw total
idx['hyperparams'].unique() # list of unique best configurations

idx.to_csv('hypeparams_state_level.csv') # save out

''' based on each state's best performing model seen in df_grid_results, create dedicated dictionaries
where to store state-level dataset based on combination of number of estimators and learning rate. 
These may need to be modified manually (the number between parentheses refer to dataset in alphabetical order,
such tath 0='Alabama', 1='Alaska', and so on). '''

# TRAIN SET


# 0.5 and 50

X_train_d_05_50 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {9, 10, 14, 15, 17, 30, 35, 45}}

Y_train_d_05_50 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {9, 10, 14, 15, 17, 30, 35, 45}}

# 0.5 and 200

X_train_d_05_200 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {0, 16, 42}}

Y_train_d_05_200 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {0, 16, 42}}


# 0.5 and 100

X_train_d_05_100 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {7, 13, 37, 44, 50}}

Y_train_d_05_100 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {7, 13, 37, 44, 50}}


# 0.3 and 50 

X_train_d_03_50 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {20, 49}}

Y_train_d_03_50 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {20, 49}}


# 0.3 and 200

X_train_d_03_200 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {3, 5, 19, 38, 46, 48}}

Y_train_d_03_200 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {3, 5, 19, 38, 46, 48}}

# 0.3 and 100


X_train_d_03_100 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {2, 21, 22, 23, 25, 29, 36, 40, 47}}

Y_train_d_03_100 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {2, 21, 22, 23, 25, 29, 36, 40, 47}}

# 0.1 and 200

X_train_d_01_200 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {4, 33, 43}}

Y_train_d_01_200 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {4, 33, 43}}


# 0.1 and 100

X_train_d_01_100 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {6, 26, 28, 31}}

Y_train_d_01_100 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {6, 26, 28, 31}}

# 0.01 and 50 

X_train_d_001_50 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {32}}

Y_train_d_001_50 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {32}}

# 0.01 and 100

X_train_d_001_100 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {18}}

Y_train_d_001_100 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {18}}


# 0.01 and 200

X_train_d_001_200 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {8, 39}}

Y_train_d_001_200 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {8, 39}}


# 0.001 and 200

X_train_d_0001_200 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {1}}

Y_train_d_0001_200 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {1}}



# 0.001 and 50
X_train_d_0001_50 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {11, 24, 34, 41}}

Y_train_d_0001_50 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {11, 24, 34, 41}}


# 0.001 and 100
X_train_d_0001_100 = {key: X_train_d[key] for key in X_train_d.keys()
                               & {12, 27}}

Y_train_d_0001_100 = {key: Y_train_d[key] for key in Y_train_d.keys()
                               & {12, 27}}

############## TEST SETS

# 0.5 and 50

X_test_d_05_50 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {9, 10, 14, 15, 17, 30, 35, 45}}

Y_test_d_05_50 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {9, 10, 14, 15, 17, 30, 35, 45}}

# 0.5 and 200

X_test_d_05_200 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {0, 16, 42}}

Y_test_d_05_200 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {0, 16, 42}}


# 0.5 and 100

X_test_d_05_100 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {7, 13, 37, 44, 50}}

Y_test_d_05_100 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {7, 13, 37, 44, 50}}


# 0.3 and 50 

X_test_d_03_50 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {20, 49}}

Y_test_d_03_50 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {20, 49}}


# 0.3 and 200

X_test_d_03_200 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {3, 5, 19, 38, 46, 48}}

Y_test_d_03_200 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {3, 5, 19, 38, 46, 48}}

# 0.3 and 100


X_test_d_03_100 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {2, 21, 22, 23, 25, 29, 36, 40, 47}}

Y_test_d_03_100 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {2, 21, 22, 23, 25, 29, 36, 40, 47}}

# 0.1 and 200

X_test_d_01_200 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {4, 33, 43}}

Y_test_d_01_200 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {4, 33, 43}}


# 0.1 and 100

X_test_d_01_100 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {6, 26, 28, 31}}

Y_test_d_01_100 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {6, 26, 28, 31}}

# 0.01 and 50 

X_test_d_001_50 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {32}}

Y_test_d_001_50 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {32}}

# 0.01 and 100

X_test_d_001_100 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {18}}

Y_test_d_001_100 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {18}}


# 0.01 and 200

X_test_d_001_200 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {8, 39}}

Y_test_d_001_200 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {8, 39}}


# 0.001 and 200

X_test_d_0001_200 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {1}}

Y_test_d_0001_200 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {1}}



# 0.001 and 50
X_test_d_0001_50 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {11, 24, 34, 41}}

Y_test_d_0001_50 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {11, 24, 34, 41}}


# 0.001 and 100
X_test_d_0001_100 = {key: X_test_d[key] for key in X_test_d.keys()
                               & {12, 27}}

Y_test_d_0001_100 = {key: Y_test_d[key] for key in Y_test_d.keys()
                               & {12, 27}}

##### cross validated models based on best hyperparm combination 
from sklearn.metrics import classification_report
scoring = ['accuracy', 'balanced_accuracy','precision', 'recall', 'f1']

# 0.5 and 50
fit05_50 = {}
clas05_50 = {}
prec05_50 = {}
bacc05_50 = {}
for key in X_train_d_05_50.keys():
    mod05_50=XGBClassifier(objective="reg:logistic", learning_rate=0.5, n_estimators=50, random_state=42)
    fit05_50[key]=mod05_50.fit(X_train_d_05_50[key], Y_train_d_05_50[key])
    clas05_50[key] = mod05_50.predict(X_test_d_05_50[key])
    bacc05_50[key] = balanced_accuracy_score(Y_test_d_05_50[key], clas05_50[key])
    prec05_50[key] = precision_score(Y_test_d_05_50[key], clas05_50[key])
    
    
# 0.5 and 200
fit05_200 = {}
clas05_200 = {}
prec05_200 = {}
bacc05_200 = {}
for key in X_train_d_05_200.keys():
    mod05_200=XGBClassifier(objective="reg:logistic", learning_rate=0.5, n_estimators=200, random_state=42)
    fit05_200[key] = mod05_200.fit(X_train_d_05_200[key], Y_train_d_05_200[key])
    clas05_200[key] = mod05_200.predict(X_test_d_05_200[key])
    bacc05_200[key] = balanced_accuracy_score(Y_test_d_05_200[key], clas05_200[key])
    prec05_200[key] = precision_score(Y_test_d_05_200[key], clas05_200[key])



# 0.5 and 100
fit05_100 = {}
clas05_100 = {}
prec05_100 = {}
bacc05_100 = {}
for key in X_train_d_05_100.keys():
    mod05_100=XGBClassifier(objective="reg:logistic", learning_rate=0.5, n_estimators=100, random_state=42)
    fit05_100[key] = mod05_100.fit(X_train_d_05_100[key], Y_train_d_05_100[key])
    clas05_100[key] = mod05_100.predict(X_test_d_05_100[key])
    bacc05_100[key] = balanced_accuracy_score(Y_test_d_05_100[key], clas05_100[key])
    prec05_100[key] = precision_score(Y_test_d_05_100[key], clas05_100[key])


# 0.3 and 50
fit03_50 = {}
clas03_50 = {}
prec03_50 = {}
bacc03_50 = {}
for key in X_train_d_03_50.keys():
    mod03_50=XGBClassifier(objective="reg:logistic", learning_rate=0.3, n_estimators=50, random_state=42)
    fit03_50[key]= mod03_50.fit(X_train_d_03_50[key], Y_train_d_03_50[key])
    clas03_50[key] = mod03_50.predict(X_test_d_03_50[key])
    bacc03_50[key] = balanced_accuracy_score(Y_test_d_03_50[key], clas03_50[key])
    prec03_50[key] = precision_score(Y_test_d_03_50[key], clas03_50[key])



# 0.3 and 200
fit03_200= {}
clas03_200 = {}
prec03_200 = {}
bacc03_200 = {}
for key in X_train_d_03_200.keys():
    mod03_200=XGBClassifier(objective="reg:logistic", learning_rate=0.3, n_estimators=200, random_state=42)
    fit03_200[key]=mod03_200.fit(X_train_d_03_200[key], Y_train_d_03_200[key])
    clas03_200[key] = mod03_200.predict(X_test_d_03_200[key])
    bacc03_200[key] = balanced_accuracy_score(Y_test_d_03_200[key], clas03_200[key])
    prec03_200[key] = precision_score(Y_test_d_03_200[key], clas03_200[key])



# 0.3 and 100
fit03_100 = {}
clas03_100 = {}
prec03_100 = {}
bacc03_100 = {}
for key in X_train_d_03_100.keys():
    mod03_100=XGBClassifier(objective="reg:logistic", learning_rate=0.3, n_estimators=100, random_state=42)
    fit03_100[key]=mod03_100.fit(X_train_d_03_100[key], Y_train_d_03_100[key])
    clas03_100[key] = mod03_100.predict(X_test_d_03_100[key])
    bacc03_100[key] = balanced_accuracy_score(Y_test_d_03_100[key], clas03_100[key])
    prec03_100[key] = precision_score(Y_test_d_03_100[key], clas03_100[key])


# 0.1 and 100
fit01_100 = {}
clas01_100 = {}
prec01_100 = {}
bacc01_100 = {}
for key in X_train_d_01_100.keys():
    mod01_100=XGBClassifier(objective="reg:logistic", learning_rate=0.1, n_estimators=100, random_state=42)
    fit01_100[key]=mod01_100.fit(X_train_d_01_100[key], Y_train_d_01_100[key])
    clas01_100[key] = mod01_100.predict(X_test_d_01_100[key])
    bacc01_100[key] = balanced_accuracy_score(Y_test_d_01_100[key], clas01_100[key])
    prec01_100[key] = precision_score(Y_test_d_01_100[key], clas01_100[key])



# 0.1 and 200
fit01_200 = {}
clas01_200 = {}
prec01_200 = {}
bacc01_200 = {}
for key in X_train_d_01_200.keys():
    mod01_200=XGBClassifier(objective="reg:logistic", learning_rate=0.1, n_estimators=200, random_state=42)
    fit01_200[key]=mod01_200.fit(X_train_d_01_200[key], Y_train_d_01_200[key])
    clas01_200[key] = mod01_200.predict(X_test_d_01_200[key])
    bacc01_200[key] = balanced_accuracy_score(Y_test_d_01_200[key], clas01_200[key])
    prec01_200[key] = precision_score(Y_test_d_01_200[key], clas01_200[key])


# 0.01 and 50
fit001_50 = {}
clas001_50 = {}
prec001_50 = {}
bacc001_50 = {}
for key in X_train_d_001_50.keys():
    mod001_50=XGBClassifier(objective="reg:logistic", learning_rate=0.01, n_estimators=50, random_state=42)
    fit001_50[key]=mod001_50.fit(X_train_d_001_50[key], Y_train_d_001_50[key])
    clas001_50[key] = mod001_50.predict(X_test_d_001_50[key])
    bacc001_50[key] = balanced_accuracy_score(Y_test_d_001_50[key], clas001_50[key])
    prec001_50[key] = precision_score(Y_test_d_001_50[key], clas001_50[key])
    

# 0.01 and 100
fit001_100 = {}
clas001_100 = {}
prec001_100 = {}
bacc001_100 = {}
for key in X_train_d_001_100.keys():
    mod001_100=XGBClassifier(objective="reg:logistic", learning_rate=0.01, n_estimators=100, random_state=42)
    fit001_100[key]=mod001_100.fit(X_train_d_001_100[key], Y_train_d_001_100[key])
    clas001_100[key] = mod001_100.predict(X_test_d_001_100[key])
    bacc001_100[key] = balanced_accuracy_score(Y_test_d_001_100[key], clas001_100[key])
    prec001_100[key] = precision_score(Y_test_d_001_100[key], clas001_100[key])


# 0.001 and 200
fit0001_200 = {}
clas0001_200 = {}
prec0001_200 = {}
bacc0001_200 = {}
for key in X_train_d_0001_200.keys():
    mod0001_200=XGBClassifier(objective="reg:logistic", learning_rate=0.001, n_estimators=200, random_state=42)
    fit0001_200[key]=mod0001_200.fit(X_train_d_0001_200[key], Y_train_d_0001_200[key])
    clas0001_200[key] = mod0001_200.predict(X_test_d_0001_200[key])
    bacc0001_200[key] = balanced_accuracy_score(Y_test_d_0001_200[key], clas0001_200[key])
    prec0001_200[key] = precision_score(Y_test_d_0001_200[key], clas0001_200[key])


# 0.01 and 200
fit001_200 = {}
clas001_200 = {}
prec001_200 = {}
bacc001_200 = {}
for key in X_train_d_001_200.keys():
    mod001_200=XGBClassifier(objective="reg:logistic", learning_rate=0.01, n_estimators=200, random_state=42)
    fit001_200[key]=mod001_200.fit(X_train_d_001_200[key], Y_train_d_001_200[key])
    clas001_200[key] = mod001_200.predict(X_test_d_001_200[key])
    bacc001_200[key] = balanced_accuracy_score(Y_test_d_001_200[key], clas001_200[key])
    prec001_200[key] = precision_score(Y_test_d_001_200[key], clas001_200[key])

    
# 0.001 and 50
fit0001_50 = {}
clas0001_50 = {}
prec0001_50 = {}
bacc0001_50 = {}
for key in X_train_d_0001_50.keys():
    mod0001_50=XGBClassifier(objective="reg:logistic", learning_rate=0.001, n_estimators=50, random_state=42)
    fit0001_50[key]=mod0001_50.fit(X_train_d_0001_50[key], Y_train_d_0001_50[key])
    clas0001_50[key] = mod0001_50.predict(X_test_d_0001_50[key])
    bacc0001_50[key] = balanced_accuracy_score(Y_test_d_0001_50[key], clas0001_50[key])
    prec0001_50[key] = precision_score(Y_test_d_0001_50[key], clas0001_50[key])

# 0.001 and 100
fit0001_100 = {}
clas0001_100 = {}
prec0001_100 = {}
bacc0001_100 = {}
for key in X_train_d_0001_100.keys():
    mod0001_100=XGBClassifier(objective="reg:logistic", learning_rate=0.001, n_estimators=100, random_state=42)
    fit0001_100[key]=mod0001_100.fit(X_train_d_0001_100[key], Y_train_d_0001_100[key])
    clas0001_100[key] = mod0001_100.predict(X_test_d_0001_100[key])
    bacc0001_100[key] = balanced_accuracy_score(Y_test_d_0001_100[key], clas0001_100[key])
    prec0001_100[key] = precision_score(Y_test_d_0001_100[key], clas0001_100[key])


##### combine performance dictionaries 
   
## precision
prec_state_models = dict(prec05_50)
prec_state_models.update(prec05_200)
prec_state_models.update(prec05_100)
prec_state_models.update(prec03_50)
prec_state_models.update(prec03_200)
prec_state_models.update(prec03_100)
prec_state_models.update(prec01_100)
prec_state_models.update(prec01_200)
prec_state_models.update(prec001_100)
prec_state_models.update(prec001_50)
prec_state_models.update(prec0001_200)
prec_state_models.update(prec001_200)
prec_state_models.update(prec0001_50)
prec_state_models.update(prec0001_100)

prec_state_df = pd.DataFrame(prec_state_models.items(), columns=['State', 'Precision'])




# balanced accuracy
bacc_state_models = dict(bacc05_50)
bacc_state_models.update(bacc05_200)
bacc_state_models.update(bacc05_100)
bacc_state_models.update(bacc03_50)
bacc_state_models.update(bacc03_200)
bacc_state_models.update(bacc03_100)
bacc_state_models.update(bacc01_100)
bacc_state_models.update(bacc01_200)
bacc_state_models.update(bacc001_50)
bacc_state_models.update(bacc001_100)
bacc_state_models.update(bacc0001_200)
bacc_state_models.update(bacc001_200)
bacc_state_models.update(bacc0001_50)
bacc_state_models.update(bacc0001_100)

bacc_state_df = pd.DataFrame(bacc_state_models.items(), columns=['State', 'Balanced Accuracy'])

# merge the two 

metrics_state_df = pd.merge(prec_state_df, bacc_state_df, on='State')
metrics_state_df = metrics_state_df.sort_values('State')

state_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 
              'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 
              'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 
              'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 
              'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 
              'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 
              'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 
              'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhodes Island', 'South Carolina', 
              'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 
              'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
state_init = ['AL','AK',	'AZ',	'AR',	'CA',	'CO',	'CT',	'DE',	'DC',	'FL',	
              'GA',	'HI',	'ID',	'IL',	'IN',	'IA',	'KS',	'KY',	'LA',	'ME',	
              'MD',	'MA',	'MI',	'MN',	'MS',	'MO',	'MT',	'NE',	'NV',	'NH',	
              'NJ',	'NM',	'NY',	'NC',	'ND',	'OH',	'OK',	'OR',	'PA',	'RI',	
              'SC',	'SD',	'TN',	'TX',	'UT',	'VT',	'VA',	'WA',	'WV',	'WI',	'WY']

metrics_state_df['StateName']=state_list
metrics_state_df['StateInit']=state_init


# visualization of performance


import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Microsoft Sans Serif"
matplotlib.rcParams['font.family'] = "sans-serif"



matplotlib.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
perfplot1 = metrics_state_df.plot.barh(y='Balanced Accuracy', 
                                                  x='StateInit', legend=False, color='maroon',ax=axes[0], alpha=0.4)
perfplot1.invert_yaxis()
perfplot1.grid()
perfplot1.set_ylabel('State')
perfplot1.set_xlabel('Balanced Accuracy')
perfplot1.set_xlim([0,1])
perfplot2 = metrics_state_df.plot.barh(y='Precision', 
                                                 x='StateInit', legend=False,color='teal',ax=axes[1], alpha=0.4)
perfplot2.invert_yaxis()
perfplot2.set_ylabel('')
perfplot2.set_xlabel('Precision')
perfplot2.grid()






