''' import again dataset and process it, without leave-one out after dummy 
feature creation, as tree-methods and particularly XGBoost 
do not suffer from the problem of perfect collinearity
NOTE: due to observation shuffling and its stochastic nature, slight 
changes in predictive as well as explainability results may occur'''


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

df_d = pd.get_dummies(df_red, columns = ['Decade', 'Month', 'Homicide', 'VicSex', 'Age_Cat', 'Weapon', 'Circumstance', 'Agentype', 'VicRace',  'Monthly_Overlap','Solved'],)

df_d.drop('ID', axis=1, inplace=True)
df_d.drop('Year', axis=1, inplace=True)
df_d.drop('VicAge', axis=1, inplace=True)   
df_d.drop(['Homicide_Murder and non-negligent manslaughter',
           'Monthly_Overlap_False','Solved_No'], axis=1, inplace=True) 

# shuffle to avoid any potential order
df_d = shuffle(df_d)

# dataset creation
X = df_d.iloc[:,0:-1]
y = df_d.iloc[:,-1]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.3, random_state=1)


# Rename columsn for better visualization

X_train.rename(columns={"VicCount":"N of Victims", "OffCount": "N of Offenders", "Decade_10s": "Decade: 2010s", "Decade_70s": "Decade: 1970s", "Decade_80s": "Decade: 1980s",
                             "Decade_90s": "Decade: 1990s",  "Decade_00s": "Decade: 2000s", "Month_August": "Month: August", "Month_December": "Month: December", "Month_February": "Month: February",
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

X_validation.rename(columns={"VicCount":"N of Victims", "OffCount": "N of Offenders", "Decade_00s": "Decade: 2000s","Decade_10s": "Decade: 2010s", "Decade_70s": "Decade: 1970s", "Decade_80s": "Decade: 1980s",
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


# saving sets for replication

USA_Xtrain_set = open("USAShap_Xtrain_set.pkl","wb")
pickle.dump(X_train,USA_Xtrain_set )
USA_Xtrain_set.close()

USA_Xvalidation_set = open("USAShap_Xvalidation_set.pkl","wb")
pickle.dump(X_validation,USA_Xvalidation_set )
USA_Xvalidation_set.close()

USA_Ytrain_set = open("USAShap_Ytrain_set.pkl","wb")
pickle.dump(Y_train,USA_Ytrain_set )
USA_Ytrain_set.close()


USA_Yvalidation_set = open("USAShap_Yvalidation_set.pkl","wb")
pickle.dump(Y_validation,USA_Yvalidation_set )
USA_Yvalidation_set.close()

# to import them for future use:
X_train = pickle.load(open("USAShap_Xtrain_set.pkl", "rb"))
X_validation =  pickle.load(open("USAShap_Xvalidation_set.pkl", "rb"))
Y_train= pickle.load(open("USAShap_Ytrain_set.pkl", "rb"))
Y_validation =  pickle.load(open("USAShap_Yvalidation_set.pkl", "rb"))

# repeat best performing model fit (XGBoost, 200 learners, 0.5 learning rate)

from sklearn.metrics import zero_one_loss


mod_best = XGBClassifier(objective="reg:logistic", random_state=42,
                         n_estimators=200, learning_rate=0.5)

mod_best.fit(X_train, Y_train)
mod_pred = mod_best.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, mod_pred))
print("Precision_Accuracy", precision_score(Y_validation, mod_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, mod_pred, normalize=False)) # to count raw number of wrongly classified observations

# comparison with second best and worst among best configurations

gbc =GradientBoostingClassifier(random_state=42,
                         n_estimators=200, learning_rate=0.5)


gbc.fit(X_train, Y_train)
gbc_pred = gbc.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, gbc_pred))
print("Precision_Accuracy", precision_score(Y_validation, gbc_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, gbc_pred, normalize=False))


lda = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
lda.fit(X_train, Y_train)
lda_pred = lda.predict(X_validation)
print("Balanced_Accuracy", balanced_accuracy_score(Y_validation, lda_pred))
print("Precision_Accuracy", precision_score(Y_validation, lda_pred))
print ("Zero one Loss", zero_one_loss(Y_validation, lda_pred, normalize=False))



''' SHAP model with best performing algorithm and configuration '''

import shap
shap.initjs()

explainer = shap.TreeExplainer(mod_best)


shap_values2 = explainer(X_validation)

'''visualization'''

# barplot
shap.summary_plot(shap_values2.values, X_validation, plot_type="bar", color='steelblue', max_display=20)
plt.xticks(fontsize=16)
plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', size=16)
plt.yticks(fontsize=16)


# summary distribution plot

shap.summary_plot(shap_values2.values, X_validation, alpha=0.05, max_display=20, show=False)
plt.xticks(fontsize=16)
plt.xlabel('SHAP value (impact on model output) - Log Odds', size=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()


# plot for local explanation for first feature in test set
shap.plots.waterfall(shap_values2[0])
