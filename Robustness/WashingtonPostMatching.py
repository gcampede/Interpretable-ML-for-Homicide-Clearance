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
from sklearn.metrics import zero_one_loss 
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
np.set_printoptions(suppress=True)
%matplotlib qt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Microsoft Sans Serif"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"


os.chdir('C:\\Users')

df = pd.read_csv('SHR76_19.csv', delimiter=',')
wp = pd.read_csv('homicide-data.csv', encoding='iso-8859-1')


#processing map
df = df[df['Year']>2006] # keep only homicides occurred from 2007 on

df.rename(columns={'Agency': 'City', 'VicAge': 'Victim_Age', 'VicSex': 'Victim_Sex'}, inplace=True)

df=df[['ID','Year', 'Month', 'City', 'Victim_Age', 'Victim_Sex', 'Solved']]

df['Year'] = df['Year'].astype(str)
df['Victim_Age'] = df['Victim_Age'].astype(str)
df['Code']=df['Year']+"-"+df['Month']+"-"+df['City']+"-"+df['Victim_Age']+"-"+df['Victim_Sex']
df['ID+Age']=df['ID']+"-"+df['Victim_Age']

# processing wp
wp['reported_date'] = wp['reported_date'].astype(str)
wp['Year'], wp['Month'] = wp['reported_date'].str[:4], wp['reported_date'].str[4:6]



di = {'01': "January", '02': "February", '03': "March", '04':"April", '05':"May", '06':"June",
      '07':"July", '08':"August", '09':"September", '10':"October", '11':"November", '12':"December"}

wp['Month']=wp['Month'].map(di)



wp['disposition'].replace('Open/No arrest', 'No', inplace=True)
wp['disposition'].replace(['Closed without arrest', 'Closed by arrest'], 'Yes', inplace=True)
wp.rename(columns={'disposition': 'Solved'}, inplace=True)
wp.rename(columns={'city': 'City', 'victim_age': 'Victim_Age', 'victim_sex': 'Victim_Sex'}, inplace=True) # rename variables

wp=wp[['Year', 'Month', 'City', 'Victim_Age', 'Victim_Sex', 'Solved']]
ncols = ['Year','Victim_Age']
wp[ncols]=wp[ncols].apply(pd.to_numeric, errors='coerce', axis=1)
wp['Year'] = wp['Year'].apply(lambda f: format(f, '.0f'))
wp['Victim_Age'] = wp['Victim_Age'].apply(lambda f: format(f, '.0f'))
wp['Code']=wp['Year']+"-"+wp['Month']+"-"+wp['City']+"-"+wp['Victim_Age']+"-"+wp['Victim_Sex']


# reset indexes

wp.reset_index(inplace=True, drop=True)
df.reset_index(inplace=True, drop=True)
wpR=wp[['Code', 'Solved']]
wpR['Source']='WP'
dfR=df[['ID','ID+Age','Code', 'Solved']]
dfR['Source']='MAP'
tR=pd.merge(wpR,dfR, how='inner', on=['Code'])
tR2 = pd.merge(left=wpR, right=dfR, on='Code', how='inner')

duplicateRowsDF = tR2[tR2.duplicated(subset=None, keep=False)] # coutn duplicates - left

tR2.drop_duplicates(subset='Code',keep=False, inplace=True) # drop duplicates
tR2.isna().sum() # count nans
tR2 = tR2.dropna() # drop nans



tR2['Num_SolvWP'] = tR2.Solved_x.apply(lambda x: 1 if x == 'Yes' else 0)
tR2['Num_SolvMAP'] = tR2.Solved_y.apply(lambda x: 1 if x == 'Yes' else 0)
comparison_column = np.where(tR2['Num_SolvWP']== tR2['Num_SolvMAP'], True, False)
tR2['equal'] = comparison_column

tR2['equal'].value_counts()
tR2['difference'] = tR2['Num_SolvWP']-tR2['Num_SolvMAP']



############################################# keep only true (equal outcomes)

tR3 = tR2[tR2['difference']==0]
tR3= tR3[['ID', 'Code']]

mapd = pd.read_csv('C:\\Users\\Gian Maria\\Desktop\\Unitn\\map\\SHR76_19.csv')
mapd['VicAge']=mapd['VicAge'].astype(str)
mapd['Year'] = mapd['Year'].astype(str)
mapd['Code'] = mapd['Year']+"-"+mapd['Month']+"-"+mapd['Agency']+"-"+mapd['VicAge']+"-"+mapd['VicSex']
mapR = pd.merge(left=tR3, right=mapd, on=['ID', 'Code'], how='inner')
duplicateRowsmapR = mapR[mapR.duplicated(subset=None, keep=False)] # count duplicates - left

mapR.drop_duplicates(keep=False, inplace=True) # drop duplicates


# keep true and switch discordant outcomes 

tR4 = tR2[['ID', 'Code', 'difference']]

mapR2 = pd.merge(left=tR4, right=mapd, on=['ID', 'Code'], how='inner')

mapR2['Solved?'] = mapR2['Solved']

mapR2['Solved']= np.where(mapR2.difference == -1, 'No', mapR2['Solved'])
mapR2['Solved']= np.where(mapR2.difference == 1, 'Yes', mapR2['Solved'])




'''dataset prep for robustness (first robustness - keep only equal outcomes)'''

''' data processing '''

#mapR.drop(mapR[(mapR['Solved'] == 'Yes') & (mapR['OffSex'] == 'Unknown')].index, inplace=True) # remove doubtful solved cases

### homicide overlap variable
mapR['Year2']=mapR['Year'].copy()
mapR['Year2']=mapR['Year2'].astype(str)
mapR['Year2'] = str('Year2')
mapR['Code_Overlap'] = mapR['State']+"_"+mapR['Agency']+"_"+mapR['Year2']+"_"+mapR['Month']
mapR['Code_Overlap+Agency'] = mapR['State']+"_"+mapR['Agency']+"_"+mapR['Agentype']+"_"+mapR['Year2']+"_"+mapR['Month']
mapR['Monthly_Overlap']= mapR.duplicated('Code_Overlap')
mapR['Monthly_Agency_Overlap']=mapR.duplicated('Code_Overlap+Agency')

### variable mapping whether the homicide involved more than 1 victim
mapR['VicCount'] = mapR['VicCount']+1
mapR['OffCount'] = mapR['OffCount']+1

mapR_red = mapR[['ID', 'Month', 'Solved', 'Homicide', 'VicAge', 'VicSex', 'Circumstance','Weapon', 'Agentype', 'VicRace', 'VicCount', 'OffCount', 'Monthly_Overlap']] #Circumstance #State


### age function


mapR_red['VicAge']=mapR_red['VicAge'].apply(pd.to_numeric, errors='coerce')
mapR_red = mapR_red[mapR_red.VicAge !=999] # remove unknown age

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

mapR_red['Age_Cat'] = mapR_red.apply(lambda x: age_cat(x['VicAge']), axis=1) 
mapR_red.sort_values('VicRace', ascending=False, inplace=True)

mapR_d = pd.get_dummies(mapR_red, columns = ['Month', 'Homicide', 
                                          'VicSex', 'Age_Cat', 'Weapon', 'Circumstance', 
                                          'Agentype', 'VicRace',  'Monthly_Overlap','Solved'], drop_first=False)


mapR_d.drop(['ID', 'VicAge', 'Homicide_Murder and non-negligent manslaughter', 'Monthly_Overlap_False','Solved_No'], axis=1, inplace=True)
mapR_d = shuffle(mapR_d)


# dataset creation
X = mapR_d.iloc[:,0:-1]
y = mapR_d.iloc[:,-1]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.3, random_state=1)

X_train.rename(columns={"VicCount":"N of Victims", "OffCount": "N of Offenders", "Month_August": "Month: August", "Month_December": "Month: December", "Month_February": "Month: February",
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

X_validation.rename(columns={"VicCount":"N of Victims", "OffCount": "N of Offenders","Month_August": "Month: August", "Month_December": "Month: December", "Month_February": "Month: February",
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


mod_best = XGBClassifier(objective="reg:logistic", random_state=42,
                         n_estimators=200, learning_rate=0.5)

mod_best.fit(X_train, Y_train)
mod_pred = mod_best.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, mod_pred))
print("Precision_Accuracy", precision_score(Y_validation, mod_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, mod_pred, normalize=False))


import shap
shap.initjs()

explainer = shap.TreeExplainer(mod_best)
shap_values = explainer(X_validation)


# barplot
shap.summary_plot(shap_values.values, X_validation, plot_type="bar", color='lightblue', max_display=15)
plt.xticks(fontsize=16)
plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', size=16)
plt.yticks(fontsize=16)
# distribution plot
fig = matplotlib.pyplot.gcf()
plt.figure(figsize=(3,4))
shap.summary_plot(shap_values.values, X_validation, alpha=0.05, max_display=15, show=False)
plt.xticks(fontsize=16)
plt.xlabel('SHAP value (impact on model output) - Log Odds', size=16)
#plt.ylabel('Most impactful features', size=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()





'second robustness: change MAP based on WP'


mapR2['Year2']=mapR2['Year'].copy()
mapR2['Year2']=mapR2['Year2'].astype(str)
mapR2['Year2'] = str('Year2')
mapR2['Code_Overlap'] = mapR2['State']+"_"+mapR2['Agency']+"_"+mapR2['Year2']+"_"+mapR2['Month']
mapR2['Code_Overlap+Agency'] = mapR2['State']+"_"+mapR2['Agency']+"_"+mapR2['Agentype']+"_"+mapR2['Year2']+"_"+mapR2['Month']
mapR2['Monthly_Overlap']= mapR2.duplicated('Code_Overlap')
mapR2['Monthly_Agency_Overlap']=mapR2.duplicated('Code_Overlap+Agency')

### variable mapping whether the homicide involved more than 1 victim
mapR2['VicCount'] = mapR2['VicCount']+1
mapR2['OffCount'] = mapR2['OffCount']+1

mapR2_red = mapR2[['ID', 'Month', 'Solved', 'Homicide', 'VicAge', 'VicSex', 'Circumstance','Weapon', 'Agentype', 'VicRace', 'VicCount', 'OffCount', 'Monthly_Overlap']] #Circumstance #State


### age function


mapR2_red['VicAge']=mapR2_red['VicAge'].apply(pd.to_numeric, errors='coerce')
mapR2_red = mapR2_red[mapR2_red.VicAge !=999] # remove unknown age

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

mapR2_red['Age_Cat'] = mapR2_red.apply(lambda x: age_cat(x['VicAge']), axis=1) 
mapR2_red.sort_values('VicRace', ascending=False, inplace=True)

mapR2_d = pd.get_dummies(mapR2_red, columns = ['Month', 'Homicide', 
                                          'VicSex', 'Age_Cat', 'Weapon', 'Circumstance', 
                                          'Agentype', 'VicRace',  'Monthly_Overlap','Solved'], drop_first=False)


mapR2_d.drop(['ID', 'VicAge', 'Homicide_Murder and non-negligent manslaughter', 'Monthly_Overlap_False','Solved_No'], axis=1, inplace=True)
mapR2_d = shuffle(mapR2_d)


# dataset creation
X2 = mapR2_d.iloc[:,0:-1]
y2 = mapR2_d.iloc[:,-1]

X_train2, X_validation2, Y_train2, Y_validation2 = train_test_split(X2, y2, test_size=0.3, random_state=1)

X_train2.rename(columns={"VicCount":"N of Victims", "OffCount": "N of Offenders", "Month_August": "Month: August", "Month_December": "Month: December", "Month_February": "Month: February",
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

X_validation2.rename(columns={"VicCount":"N of Victims", "OffCount": "N of Offenders","Month_August": "Month: August", "Month_December": "Month: December", "Month_February": "Month: February",
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


mod_best2 = XGBClassifier(objective="reg:logistic", random_state=42,
                         n_estimators=200, learning_rate=0.5)

mod_best2.fit(X_train2, Y_train2)
mod_pred2 = mod_best2.predict(X_validation2)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation2, mod_pred2))
print("Precision_Accuracy", precision_score(Y_validation2, mod_pred2))
print ("Zero one Loss", zero_one_loss(Y_validation2, mod_pred2, normalize=False))


import shap
shap.initjs()

explainer2 = shap.TreeExplainer(mod_best2)
shap_values2 = explainer2(X_validation2)


# barplot
shap.summary_plot(shap_values2.values, X_validation2, plot_type="bar", color='lightblue', max_display=15)
plt.xticks(fontsize=16)
plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', size=16)
plt.yticks(fontsize=16)
# distribution plot
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15.5, 15.5)
shap.summary_plot(shap_values2.values, X_validation2, alpha=0.05, max_display=15, show=False)
plt.xticks(fontsize=16)
plt.xlabel('SHAP value (impact on model output) - Log Odds', size=16)
#plt.ylabel('Most impactful features', size=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
