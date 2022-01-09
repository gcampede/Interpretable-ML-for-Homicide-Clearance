''' this script should be used after having run the 'state_wise_prediction.py' script, so that 
relevant data is already in memory. Otherwise, to upload state-wise train and test datasets,
used pickle as indicated below'''

#upload them
X_train_d = pickle.load(open("stateSHAP_Xtrain_sets.pkl", "rb"))
X_test_d =  pickle.load(open("stateSHAP_Xtest_sets.pkl", "rb"))
Y_train_d = pickle.load(open("stateSHAP_Ytrain_sets.pkl", "rb"))
Y_test_d =  pickle.load(open("stateSHAP_Ytest_sets.pkl", "rb"))      
    
 
 
# perform state-wise SHAP models

import shap
shap.initjs()


# 0.5 and 50 
expl05_50 = {}
shap_value05_50 = {}
for key in X_train_d_05_50.keys():
    expl05_50[key] = shap.TreeExplainer(fit05_50[key])
    shap_value05_50[key]=expl05_50[key].shap_values(X_test_d_05_50[key].values)


# 0.5 and 200 
expl05_200 = {}
shap_value05_200 = {}
for key in X_train_d_05_200.keys():
    expl05_200[key] = shap.TreeExplainer(fit05_200[key])
    shap_value05_200[key]=expl05_200[key].shap_values(X_test_d_05_200[key].values)


# 0.5 and 100 
expl05_100 = {}
shap_value05_100 = {}
for key in X_train_d_05_100.keys():
    expl05_100[key] = shap.TreeExplainer(fit05_100[key])
    shap_value05_100[key]=expl05_100[key].shap_values(X_test_d_05_100[key].values)
    
    
# 0.3 and 50 
expl03_50 = {}
shap_value03_50 = {}
for key in X_train_d_03_50.keys():
    expl03_50[key] = shap.TreeExplainer(fit03_50[key])
    shap_value03_50[key]=expl03_50[key].shap_values(X_test_d_03_50[key].values)
    
# 0.3 and 200 
expl03_200 = {}
shap_value03_200 = {}
for key in X_train_d_03_200.keys():
    expl03_200[key] = shap.TreeExplainer(fit03_200[key])
    shap_value03_200[key]=expl03_200[key].shap_values(X_test_d_03_200[key].values)


# 0.3 and 100 
expl03_100 = {}
shap_value03_100 = {}
for key in X_train_d_03_100.keys():
    expl03_100[key] = shap.TreeExplainer(fit03_100[key])
    shap_value03_100[key]=expl03_100[key].shap_values(X_test_d_03_100[key].values)


# 0.1 and 200 
expl01_200 = {}
shap_value01_200 = {}
for key in X_train_d_01_200.keys():
    expl01_200[key] = shap.TreeExplainer(fit01_200[key])
    shap_value01_200[key]=expl01_200[key].shap_values(X_test_d_01_200[key].values)
    

# 0.1 and 100 
expl01_100 = {}
shap_value01_100 = {}
for key in X_train_d_01_100.keys():
    expl01_100[key] = shap.TreeExplainer(fit01_100[key])
    shap_value01_100[key]=expl01_100[key].shap_values(X_test_d_01_100[key].values)


# 0.01 and 50 
expl001_50 = {}
shap_value001_50 = {}
for key in X_train_d_001_50.keys():
    expl001_50[key] = shap.TreeExplainer(fit001_50[key])
    shap_value001_50[key]=expl001_50[key].shap_values(X_test_d_001_50[key].values)
    

# 0.01 and 100 
expl001_100 = {}
shap_value001_100 = {}
for key in X_train_d_001_100.keys():
    expl001_100[key] = shap.TreeExplainer(fit001_100[key])
    shap_value001_100[key]=expl001_100[key].shap_values(X_test_d_001_100[key].values)


# 0.001 and 50 
expl0001_50 = {}
shap_value0001_50 = {}
for key in X_train_d_0001_50.keys():
    expl0001_50[key] = shap.TreeExplainer(fit0001_50[key])
    shap_value0001_50[key]=expl0001_50[key].shap_values(X_test_d_0001_50[key].values)
    

# 0.001 and 100 
expl0001_100 = {}
shap_value0001_100 = {}
for key in X_train_d_0001_100.keys():
    expl0001_100[key] = shap.TreeExplainer(fit0001_100[key])
    shap_value0001_100[key]=expl0001_100[key].shap_values(X_test_d_0001_100[key].values)


# 0.001 and 200 
expl0001_200 = {}
shap_value0001_200 = {}
for key in X_train_d_0001_200.keys():
    expl0001_200[key] = shap.TreeExplainer(fit0001_200[key])
    shap_value0001_200[key]=expl0001_200[key].shap_values(X_test_d_0001_200[key].values)

# 0.01 and 200 
expl001_200 = {}
shap_value001_200 = {}
for key in X_train_d_001_200.keys():
    expl001_200[key] = shap.TreeExplainer(fit001_200[key])
    shap_value001_200[key]=expl001_200[key].shap_values(X_test_d_001_200[key].values)
    
    
# create unique dictionary with arrays of shape values first
all_state_shap_dict = dict(shap_value05_50)
all_state_shap_dict.update(shap_value05_200)
all_state_shap_dict.update(shap_value05_100)
all_state_shap_dict.update(shap_value03_50)
all_state_shap_dict.update(shap_value03_200)
all_state_shap_dict.update(shap_value03_100)
all_state_shap_dict.update(shap_value01_200)
all_state_shap_dict.update(shap_value01_100)
all_state_shap_dict.update(shap_value001_50)
all_state_shap_dict.update(shap_value001_100)
all_state_shap_dict.update(shap_value0001_50)
all_state_shap_dict.update(shap_value0001_100)
all_state_shap_dict.update(shap_value001_200)
all_state_shap_dict.update(shap_value0001_200)

# also create unique dictionary for test sets to match columns 

unique_train_state = dict(X_test_d_05_50)
unique_train_state.update(X_test_d_05_200)
unique_train_state.update(X_test_d_05_100)
unique_train_state.update(X_test_d_03_50)
unique_train_state.update(X_test_d_03_200)
unique_train_state.update(X_test_d_03_100)
unique_train_state.update(X_test_d_01_200)
unique_train_state.update(X_test_d_01_100)
unique_train_state.update(X_test_d_001_50)
unique_train_state.update(X_test_d_001_100)
unique_train_state.update(X_test_d_0001_50)
unique_train_state.update(X_test_d_0001_100)
unique_train_state.update(X_test_d_0001_200)
unique_train_state.update(X_test_d_001_200)





# cretae unique shap dictionary with all state dataframes and values
shap_all_state_df = {}
for key in all_state_shap_dict:
    shap_all_state_df[key]=pd.DataFrame(data= all_state_shap_dict[key],
                                    columns=list(unique_train_state[key].columns))
    

# dataframes creation for analysis

df_shap_states = pd.concat(shap_all_state_df, axis=0).reset_index(level=0).rename({'level_0':'key'}, axis=1) # all SHAP values for all observations in the state-wise datasets concatenated as one

shap_mean = df_shap_states.mean()
abs_shap_states = df_shap_states.abs() # transform entire df_shap_states in absolute terms
shap_group= df_shap_states.groupby('key').mean() # obtain mean values (in original terms) aggregated by key


abs_shap_group = abs_shap_states.groupby('key').mean() # obtain mean values (in absolute terms) aggregated by key 
abs_shap_group.reset_index(inplace=True)
abs_shap_group['key'].astype(int)
abs_shap_group['key'].round()
abs_shap_group.set_index('key', inplace=True)

abs_shap_state_average=abs_shap_group.mean()


##### visualization
# obtain features with highest absolute values on average by state dataset. Check if fitting the model preserves all elements in this list. Slight changes may occur.
top_abs_shapmean = abs_shap_group[['Circumstance: Undetermined',
                                  'Circumstance: Other Arguments',
                                  'N of Offenders',
                                  'Decade: 1980s',
                                  'Decade: 2010s',
                                  'Victim Sex: Female',
                                  'Weapon: Handgun',
                                  'Weapon: Firearm',
                                  'Monthly State/Agency Overlap',
                                  'Decade: 1990s',
                                  'Weapon: Knife or Cutting Inst.',
                                  'Agency: Municipal Police',
                                  'Circumstance: Other',
                                  'N of Victims',
                                  'Decade_00s',
                                  'Victim Race: Black',
                                  'Circumstance: Robbery',
                                  'Decade: 1970s',
                                  'Weapon: Other/Unknown',
                                  'Circumstance: All Susp. Felony Type']]


sns.boxplot(data=top_abs_shapmean, orient='h', color='white', showfliers = False)
sns.stripplot(data=top_abs_shapmean, orient='h', color='red', size=4, alpha=0.3)


# plot summary distribution SHAP plots for the six states with the highest number of homicides
plt.figure(figsize=(10,5))
plt.subplot(3,2,1)
plt.gca().set_title('California')
shap.summary_plot(shap_value01_200[4],X_test_d_01_200[4], alpha=0.05, max_display=10,  show=False)
plt.subplot(3,2,5)
plt.gca().set_title('New York')
shap.summary_plot(shap_value001_50[32],X_test_d_001_50[32], alpha=0.05, max_display=10, show=False)
plt.subplot(3,2,4)
plt.gca().set_title('Michigan')
shap.summary_plot(shap_value03_100[22],X_test_d_03_100[22], alpha=0.05, max_display=10)
plt.subplot(3,2,2)
plt.gca().set_title('Florida')
shap.summary_plot(shap_value05_50[9],X_test_d_05_50[9], alpha=0.05, max_display=10)
plt.subplot(3,2,3)
plt.gca().set_title('Illinois')
shap.summary_plot(shap_value05_100[13],X_test_d_05_100[13], alpha=0.05, max_display=10)
plt.subplot(3,2,6)
plt.gca().set_title('Texas')
shap.summary_plot(shap_value01_200[43],X_test_d_01_200[43], alpha=0.05, max_display=10)
plt.tight_layout()
plt.show()

plt.ylabel('Features with highest average  mean(|SHAP value|) ')
plt.xlabel('State-wise mean(|SHAP value|) - Log Odds')
plt.grid()

# plot four states with highest racial disparity against Black victims

plt.figure(figsize=(10,5))
plt.subplot(2,2,2)
plt.gca().set_title('Kansas')
shap.summary_plot(shap_value05_200[16],X_test_d_05_200[16], alpha=0.05, max_display=10,  show=False)
plt.subplot(2,2,3)
plt.gca().set_title('Minnesota')
shap.summary_plot(shap_value03_100[23],X_test_d_03_100[23], alpha=0.05, max_display=10)
plt.subplot(2,2,4)
plt.gca().set_title('Missouri')
shap.summary_plot(shap_value03_100[25],X_test_d_03_100[25], alpha=0.05, max_display=10)
plt.subplot(2,2,1)
plt.gca().set_title('Iowa')
shap.summary_plot(shap_value05_50[15],X_test_d_05_50[15], alpha=0.05, max_display=10)
plt.tight_layout()
plt.show()


