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
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
np.set_printoptions(suppress=True)
%matplotlib qt
import matplotlib


df = pd.read_csv('SHR76_19.csv', delimiter=',')


''' data processing '''


### homicide overlap variable
df['Year2']=df['Year'].copy()
df['Year2']=df['Year2'].astype(str)
df['Year2'] = str('Year2')
df['Code_Overlap'] = df['State']+"_"+df['Agency']+"_"+df['Year2']+"_"+df['Month']
df['Code_Overlap+Agency'] = df['State']+"_"+df['Agency']+"_"+df['Agentype']+"_"+df['Year2']+"_"+df['Month']
df['Monthly_Overlap']= df.duplicated('Code_Overlap')
df['Monthly_Agency_Overlap']=df.duplicated('Code_Overlap+Agency')

### variable mapping whether the homicide involved more than 1 victim
df['VicCount'] = df['VicCount']+1
df['OffCount'] = df['OffCount']+1

df_red = df[['ID', 'Year', 'Month', 'Solved', 'Homicide', 'VicAge', 'VicSex', 'Circumstance','Weapon', 'Agentype', 'VicRace', 'VicCount', 'OffCount', 'Monthly_Overlap']] #Circumstance #State

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


### age function

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
df_red.sort_values('VicRace', ascending=False, inplace=True)

df_d = pd.get_dummies(df_red, columns = ['Decade', 'Month', 'Homicide', 
                                          'VicSex', 'Age_Cat', 'Weapon', 'Circumstance', 
                                          'Agentype', 'VicRace',  'Monthly_Overlap','Solved'], drop_first=False)


df_d.drop(['Year', 'ID', 'VicAge', 'Homicide_Murder and non-negligent manslaughter', 'Monthly_Overlap_False','Solved_No'], axis=1, inplace=True)
df_d = shuffle(df_d) # shuffle to avoid temporal ordering/covariate shift

# dataset creation
X = df_d.iloc[:,0:-1]
y = df_d.iloc[:,-1]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.3, random_state=1)


# Rename columsn for better visualization

X_train.rename(columns={"VicCount":"N of Victims", "OffCount": "N of Offenders", "Decade_10s": "Decade: 2010s", "Decade_70s": "Decade: 1970s", "Decade_80s": "Decade: 1980s",
                             "Decade_90s": "Decade: 1990s", "Decade_00s": "Decade:2000s", "Month_August": "Month: August", "Month_December": "Month: December", "Month_February": "Month: February",
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

X_validation.rename(columns={"VicCount":"N of Victims", "OffCount": "N of Offenders", "Decade_10s": "Decade: 2010s", "Decade_70s": "Decade: 1970s", "Decade_80s": "Decade: 1980s",
                             "Decade_90s": "Decade: 1990s", "Decade_00s": "Decade:2000s","Month_August": "Month: August", "Month_December": "Month: December", "Month_February": "Month: February",
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


# saving sets
import pickle
USA_Xtrain_set = open("USA_Xtrain_set.pkl","wb")
pickle.dump(X_train,USA_Xtrain_set )
USA_Xtrain_set.close()

USA_Xvalidation_set = open("USA_Xvalidation_set.pkl","wb")
pickle.dump(X_validation,USA_Xvalidation_set )
USA_Xvalidation_set.close()

USA_Ytrain_set = open("USA_Ytrain_set.pkl","wb")
pickle.dump(Y_train,USA_Ytrain_set )
USA_Ytrain_set.close()


USA_Yvalidation_set = open("USA_Yvalidation_set.pkl","wb")
pickle.dump(Y_validation,USA_Yvalidation_set )
USA_Yvalidation_set.close()

#upload them
X_train = pickle.load(open("USA_Xtrain_set.pkl", "rb"))
X_validation =  pickle.load(open("USA_Xvalidation_set.pkl", "rb"))
Y_train= pickle.load(open("USA_Ytrain_set.pkl", "rb"))
Y_validation =  pickle.load(open("USA_Yvalidation_set.pkl", "rb"))


#freeing up memory

del df
del df_red
del df_d


'''grid search models'''


########################### lasso, ridge regression

lr_model = LogisticRegression(random_state=42)
cv = StratifiedKFold(n_splits=5, random_state=1)
lr_space= dict()
lr_space['solver'] = ['newton-cg',  'liblinear', 'saga'] # 'lbfgs',
lr_space['penalty'] = ['l1', 'l2']
lr_space['C']= [0.01, 0.5, 1, 5, 10, 50]

scoring = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']
lr_search = GridSearchCV(lr_model, lr_space, scoring=scoring, n_jobs=5, cv=cv, verbose= 1, refit='balanced_accuracy')

lr_result = lr_search.fit(X_train, Y_train)
# summarize result
print('Best Score: %s' % lr_result.best_score_)
print('Best Hyperparameters: %s' % lr_result.best_params_)


#save object
joblib.dump(lr_search, 'lr_search.pkl')

## save dataset of results, ranked
lr_results_df = pd.DataFrame(lr_search.cv_results_)
#results_df = results_df.sort_values(by=['rank_test_score'])
lr_results_df = (
    lr_results_df
    .set_index(lr_results_df["params"].apply(
        lambda x: "_".join(str(val) for val in x.values()))
    )
    .rename_axis('kernel')
)
#lr_results_df[
#    ['params', 'rank_test_score', 'mean_test_score', 'std_test_score']
#]

lr_results_df.to_csv('log_test1.csv')


############# elastic net

ela_model = LogisticRegression(random_state=42)
cv = StratifiedKFold(n_splits=5, random_state=1)
ela_space = dict()
ela_space['solver']=['saga']
ela_space['penalty']=['elasticnet']
ela_space['l1_ratio']=[0.5]
ela_space['C']=[0.01, 0.5, 1, 5, 10, 50]

ela_search = GridSearchCV(ela_model, ela_space, scoring=scoring, n_jobs=4, cv=cv, verbose= 1, refit='balanced_accuracy')

ela_result = ela_search.fit(X_train, Y_train)
# summarize result
print('Best Score: %s' % ela_result.best_score_)
print('Best Hyperparameters: %s' % ela_result.best_params_)
joblib.dump(ela_search, 'ela_search.pkl')

## save dataset of results, ranked
ela_results_df = pd.DataFrame(ela_search.cv_results_)
#results_df = results_df.sort_values(by=['rank_test_score'])
ela_results_df = (
    ela_results_df
    .set_index(ela_results_df["params"].apply(
        lambda x: "_".join(str(val) for val in x.values()))
    )
    .rename_axis('kernel')
)


ela_results_df.to_csv('ela_test1.csv')

############################  Random forest

rf_cv = StratifiedKFold(n_splits=5, random_state=1)
rf_model = RandomForestClassifier(random_state=42)
rf_space = dict()
rf_space['n_estimators']=[50, 100, 200]
rf_space['max_depth']=[5,10,20,50,100]
rf_space['criterion']=['gini', 'entropy']

rf_search= GridSearchCV(rf_model, rf_space, scoring=scoring, cv=rf_cv, n_jobs=5, verbose= 1, refit='balanced_accuracy')
rf_result = rf_search.fit(X_train, Y_train)


#save object
joblib.dump(rf_search, 'rf_search.pkl')

#save df
rf_result_df = pd.DataFrame(rf_result.cv_results_)

rf_result_df = (
    rf_result_df
    .set_index(rf_result_df["params"].apply(
        lambda x: "_".join(str(val) for val in x.values()))
        )
        .rename_axis('kernel')
)

rf_result_df.to_csv('rf_test1.csv')



########################### gradient boosting machine

gdb_cv = StratifiedKFold(n_splits=5, random_state=1)
gdb_model = GradientBoostingClassifier(random_state=42)
gdb_space = dict()
gdb_space['n_estimators']=[50, 100, 200]
gdb_space['learning_rate']= [0.001, 0.01, 0.1, 0.3, 0.5]

gdb_search = GridSearchCV(gdb_model, gdb_space, scoring=scoring, cv=gdb_cv, n_jobs=6, verbose=2, refit = 'balanced_accuracy')
gdb_result = gdb_search.fit(X_train, Y_train)


#save object
joblib.dump(gdb_search, 'gdb_search.pkl')

#save df
gdb_result_df = pd.DataFrame(gdb_result.cv_results_)
gdb_result_df = (
    gdb_result_df
    .set_index(gdb_result_df["params"].apply(
        lambda x: "_".join(str(val) for val in x.values()))
    )
    .rename_axis('kernel')
)

gdb_result_df.to_csv('gdb_test1.csv')


#################### SVM

svm_cv = StratifiedKFold(n_splits=5, random_state=1)
svm_model = LinearSVC(dual=False)
svm_space = dict()
svm_space['penalty']=['l1', 'l2']
svm_space['C']=[0.01, 0.5, 1, 5, 10, 50]

svm_search = GridSearchCV(svm_model, svm_space, scoring=scoring, cv=svm_cv, n_jobs = 6, verbose=2, refit = 'balanced_accuracy')
svm_result = svm_search.fit(X_train, Y_train)

#save object
joblib.dump(svm_search, 'svm_search.pkl')

#save df

svm_result_df = pd.DataFrame(svm_result.cv_results_)
svm_result_df = (
    svm_result_df
    .set_index(svm_result_df["params"].apply(
        lambda x: "_".join(str(val) for val in x.values()))
    )
    .rename_axis('kernel')
)

svm_result_df.to_csv('svm_test1.csv')



######################### XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

xgb_cv = StratifiedKFold(n_splits=5, random_state=1)
xgb_model = XGBClassifier(objective="reg:logistic", random_state=42)
xgb_space = dict()
xgb_space['n_estimators']=[50, 100, 200]
xgb_space['learning_rate']= [ 0.01, 0.1, 0.3, 0.5]
xgb_space['gamma']= [0, 0.5, 1]
# xgb_space['reg_alpha']=[None, 0.1, 0.5]
# xgb_space['reg_lambda']=[None, 0.1, 0.5]

xgb_search = GridSearchCV(xgb_model, xgb_space, scoring = scoring, cv=xgb_cv, n_jobs=7, verbose=2, refit = 'balanced_accuracy')
xgb_result = xgb_search.fit(X_train, Y_train)


#save object
joblib.dump(xgb_search, 'xgb_search.pkl')

#save df
xgb_result_df = pd.DataFrame(xgb_result.cv_results_)
xgb_result_df = (
    xgb_result_df
    .set_index(xgb_result_df["params"].apply(
        lambda x: "_".join(str(val) for val in x.values()))
        )
    .rename_axis('kernel')
)

xgb_result_df.to_csv('xgb_test1.csv')


########################### lda

lda_cv = StratifiedKFold(n_splits=5, random_state=1)
lda_model = LinearDiscriminantAnalysis()
lda_space = dict()
lda_space['solver'] = ['svd', 'lsqr', 'eigen']
lda_space['shrinkage'] = ['None', 'auto']

lda_search = GridSearchCV(lda_model, lda_space, scoring=scoring, cv=lda_cv, n_jobs=5, verbose=2, refit = 'balanced_accuracy')
                        

lda_result = lda_search.fit(X_train, Y_train)

#save object
joblib.dump(lda_search, 'lda_search.pkl')

#save df
lda_result_df = pd.DataFrame(lda_result.cv_results_)
lda_result_df = (
    lda_result_df
    .set_index(lda_result_df["params"].apply(
        lambda x: "_".join(str(val) for val in x.values()))
        )
    .rename_axis('kernel')
)

lda_result_df.to_csv('lda_test1.csv')


############### decision tree

dtr_cv = StratifiedKFold(n_splits=5, random_state=1)
dtr_model = DecisionTreeClassifier(random_state=42)
dtr_space = dict()
dtr_space['max_depth']=[5,10,20,50,100]
dtr_space['criterion']=['gini', 'entropy']

dtr_search = GridSearchCV(dtr_model, dtr_space, scoring=scoring, cv=dtr_cv, n_jobs=5, verbose=2, refit = 'balanced_accuracy')

dtr_result = dtr_search.fit(X_train, Y_train)


#save object
joblib.dump(dtr_search, 'dtr_search.pkl')

#save df
dtr_result_df = pd.DataFrame(dtr_result.cv_results_)
dtr_result_df = (
    dtr_result_df
    .set_index(dtr_result_df["params"].apply(
        lambda x: "_".join(str(val) for val in x.values()))
        )
    .rename_axis('kernel')
)

dtr_result_df.to_csv('dtr_test1.csv')



############################
##### go predict with best 
from sklearn.metrics import zero_one_loss 
print('Best Score: %s' % xgb_result.best_score_)
print('Best Hyperparameters: %s' % xgb_result.best_params_)


mod_best = XGBClassifier(objective="reg:logistic", random_state=42,
                         n_estimators=200, learning_rate=0.5)

mod_best.fit(X_train, Y_train)
mod_pred = mod_best.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, mod_pred))
print("Precision_Accuracy", precision_score(Y_validation, mod_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, mod_pred, normalize=False))

#gbm
gbmB =GradientBoostingClassifier(random_state=42,
                         n_estimators=200, learning_rate=0.5)


gbmB.fit(X_train, Y_train)
gbmB_pred = gbmB.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, gbmB_pred))
print("Precision_Accuracy", precision_score(Y_validation, gbmB_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, gbmB_pred, normalize=False))


# ridge
ridgeB = LogisticRegression(random_state=42, penalty='l2', C=50, solver='liblinear')
ridgeB.fit(X_train, Y_train)
ridgeB_pred = ridgeB.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, ridgeB_pred))
print("Precision_Accuracy", precision_score(Y_validation, ridgeB_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, ridgeB_pred, normalize=False))



# lasso
lassoB = LogisticRegression(random_state=42, penalty='l1', C=50, solver='saga')
lassoB.fit(X_train, Y_train)
lassoB_pred = lassoB.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, lassoB_pred))
print("Precision_Accuracy", precision_score(Y_validation, lassoB_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, lassoB_pred, normalize=False))


# ela
elaB = LogisticRegression(random_state=42, solver='saga', C=50, l1_ratio=0.5)
elaB.fit(X_train, Y_train)
elaB_pred = elaB.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, elaB_pred))
print("Precision_Accuracy", precision_score(Y_validation, elaB_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, elaB_pred, normalize=False))



# rf
rfB = RandomForestClassifier(random_state=42, criterion='gini', max_depth=50, n_estimators=200)
rfB.fit(X_train, Y_train)
rfB_pred = rfB.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, rfB_pred))
print("Precision_Accuracy", precision_score(Y_validation, rfB_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, rfB_pred, normalize=False))


# dtr
dtrB = DecisionTreeClassifier(criterion='gini', max_depth=20, random_state=42)
dtrB.fit(X_train, Y_train)
dtrB_pred = dtrB.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, dtrB_pred))
print("Precision_Accuracy", precision_score(Y_validation, dtrB_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, dtrB_pred, normalize=False))


# svm
svmB = LinearSVC(dual=False, penalty='l2', C=50)
svmB.fit(X_train, Y_train)
svmB_pred = svmB.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, svmB_pred))
print("Precision_Accuracy", precision_score(Y_validation, svmB_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, svmB_pred, normalize=False))



#lda
ldaB = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
ldaB.fit(X_train, Y_train)
ldaB_pred = ldaB.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, ldaB_pred))
print("Precision_Accuracy", precision_score(Y_validation, ldaB_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, ldaB_pred, normalize=False))



