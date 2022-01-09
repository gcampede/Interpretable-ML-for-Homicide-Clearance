import pandas as pd, numpy as np, os, seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Microsoft Sans Serif"
matplotlib.rcParams['font.family'] = "sans-serif"

%matplotlib qt


log = pd.read_csv('log_test1.csv', delimiter=',')
ridge = log[log['kernel'].str.contains('l2')]
lasso = log[log['kernel'].str.contains('l1')]

ela = pd.read_csv('ela_test1.csv', delimiter=',')

rf = pd.read_csv('rf_test1.csv', delimiter=',')

svm = pd.read_csv('svm_test1.csv', delimiter=',')

lda = pd.read_csv('lda_test1.csv', delimiter=',')

gbm = pd.read_csv('gdb_test1.csv', delimiter=',')

xgb = pd.read_csv('xgb_test1.csv', delimiter=',')

dtr = pd.read_csv('dtr_test1.csv', delimiter = ',')

# add column

ridge['algorithm'] = 'Ridge'
lasso['algorithm'] = 'LASSO'
ela['algorithm'] = 'EN'
rf['algorithm'] = 'RF'
svm['algorithm'] = 'SVM'
lda['algorithm'] = 'LDA'
gbm['algorithm'] = 'GBM'
xgb['algorithm'] = 'XGB'
dtr['algorithm'] = 'DT'

### Recreate Subfigure 2b
########## select best configuration for each algorithm
ridge_max = ridge.loc[ridge['mean_test_balanced_accuracy'] == ridge['mean_test_balanced_accuracy'].max()]
lasso_max = lasso.loc[lasso['mean_test_balanced_accuracy'] == lasso['mean_test_balanced_accuracy'].max()]
ela_max = ela.loc[ela['mean_test_balanced_accuracy'] == ela['mean_test_balanced_accuracy'].max()]
rf_max = rf.loc[rf['mean_test_balanced_accuracy'] == rf['mean_test_balanced_accuracy'].max()]
svm_max = svm.loc[svm['mean_test_balanced_accuracy'] == svm['mean_test_balanced_accuracy'].max()]
lda_max = lda.loc[lda['mean_test_balanced_accuracy'] == lda['mean_test_balanced_accuracy'].max()]
gbm_max = gbm.loc[gbm['mean_test_balanced_accuracy'] == gbm['mean_test_balanced_accuracy'].max()]
xgb_max = xgb.loc[xgb['mean_test_balanced_accuracy'] == xgb['mean_test_balanced_accuracy'].max()]
dtr_max = dtr.loc[dtr['mean_test_balanced_accuracy'] == dtr['mean_test_balanced_accuracy'].max()]

# subset dataframes based on cols we are interested in 
ridge_max = ridge_max[['algorithm', 'kernel', 'split0_test_balanced_accuracy',
                       'split1_test_balanced_accuracy', 'split2_test_balanced_accuracy',
                       'split3_test_balanced_accuracy', 'split4_test_balanced_accuracy',
                       'split0_test_precision',
                       'split1_test_precision', 'split2_test_precision',
                       'split3_test_precision', 'split4_test_precision']]

lasso_max = lasso_max[['algorithm', 'kernel', 'split0_test_balanced_accuracy',
                       'split1_test_balanced_accuracy', 'split2_test_balanced_accuracy',
                       'split3_test_balanced_accuracy', 'split4_test_balanced_accuracy',
                       'split0_test_precision',
                       'split1_test_precision', 'split2_test_precision',
                       'split3_test_precision', 'split4_test_precision']]

ela_max = ela_max[['algorithm', 'kernel', 'split0_test_balanced_accuracy',
                       'split1_test_balanced_accuracy', 'split2_test_balanced_accuracy',
                       'split3_test_balanced_accuracy', 'split4_test_balanced_accuracy',
                       'split0_test_precision',
                       'split1_test_precision', 'split2_test_precision',
                       'split3_test_precision', 'split4_test_precision']]

rf_max = rf_max[['algorithm', 'kernel', 'split0_test_balanced_accuracy',
                       'split1_test_balanced_accuracy', 'split2_test_balanced_accuracy',
                       'split3_test_balanced_accuracy', 'split4_test_balanced_accuracy',
                       'split0_test_precision',
                       'split1_test_precision', 'split2_test_precision',
                       'split3_test_precision', 'split4_test_precision']]



svm_max = svm_max[['algorithm', 'kernel', 'split0_test_balanced_accuracy',
                       'split1_test_balanced_accuracy', 'split2_test_balanced_accuracy',
                       'split3_test_balanced_accuracy', 'split4_test_balanced_accuracy',
                       'split0_test_precision',
                       'split1_test_precision', 'split2_test_precision',
                       'split3_test_precision', 'split4_test_precision']]

lda_max = lda_max[['algorithm', 'kernel', 'split0_test_balanced_accuracy',
                       'split1_test_balanced_accuracy', 'split2_test_balanced_accuracy',
                       'split3_test_balanced_accuracy', 'split4_test_balanced_accuracy',
                       'split0_test_precision',
                       'split1_test_precision', 'split2_test_precision',
                       'split3_test_precision', 'split4_test_precision']]

gbm_max = gbm_max[['algorithm', 'kernel', 'split0_test_balanced_accuracy',
                       'split1_test_balanced_accuracy', 'split2_test_balanced_accuracy',
                       'split3_test_balanced_accuracy', 'split4_test_balanced_accuracy',
                       'split0_test_precision',
                       'split1_test_precision', 'split2_test_precision',
                       'split3_test_precision', 'split4_test_precision']]


xgb_max = xgb_max[['algorithm', 'kernel', 'split0_test_balanced_accuracy',
                       'split1_test_balanced_accuracy', 'split2_test_balanced_accuracy',
                       'split3_test_balanced_accuracy', 'split4_test_balanced_accuracy',
                       'split0_test_precision',
                       'split1_test_precision', 'split2_test_precision',
                       'split3_test_precision', 'split4_test_precision']]

dtr_max = dtr_max[['algorithm', 'kernel', 'split0_test_balanced_accuracy',
                       'split1_test_balanced_accuracy', 'split2_test_balanced_accuracy',
                       'split3_test_balanced_accuracy', 'split4_test_balanced_accuracy',
                       'split0_test_precision',
                       'split1_test_precision', 'split2_test_precision',
                       'split3_test_precision', 'split4_test_precision']]

max_frames = [ ridge_max, lasso_max, ela_max, dtr_max, rf_max, gbm_max, svm_max, xgb_max, lda_max ]

max_frame = pd.DataFrame()

for df in max_frames:
    max_frame = max_frame.append(df)
    
max_frame = max_frame[max_frame.kernel != 'auto_eigen'] # eliminate redundant lDA
max_frame_acc = pd.melt(max_frame, id_vars='algorithm', value_vars=['split0_test_balanced_accuracy',
                       'split1_test_balanced_accuracy', 'split2_test_balanced_accuracy',
                       'split3_test_balanced_accuracy', 'split4_test_balanced_accuracy'])

max_frame_pre = pd.melt(max_frame, id_vars='algorithm', value_vars=['split0_test_precision',
                       'split1_test_precision', 'split2_test_precision',
                       'split3_test_precision', 'split4_test_precision'])


boxprops = dict(linestyle='-', linewidth=3, color='white')
medianprops = dict(linestyle='-', linewidth=2.5, color='red')
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='black')
meanlineprops = dict(linestyle='-', linewidth=2.5, color='red')

whiskerprops=dict(linewidth=1.5, color='red')

f, axes = plt.subplots(1,2)
acp=sns.boxplot(data=max_frame_acc, x="algorithm", y="value",
                          ax=axes[0])
acp.set_ylabel('Balanced Accuracy', size=18)
acp.set_xlabel('Algorithm (Best Conf.)', size=18)
acp.tick_params(axis='both', labelsize=16)
acp.set_xticklabels(acp.get_xticklabels(), rotation=90)
acp.set_ylim(0.735, 0.770)
plt.setp(acp.artists, edgecolor = 'k', facecolor='w')
plt.setp(acp.lines, color='k')
myboxa = acp.artists[7]
# Change the appearance of that box
myboxa.set_facecolor('white')
myboxa.set_edgecolor('red')
for line in acp.get_lines()[41:47]:
   line.set_color('red')
prp=sns.boxplot(data=max_frame_pre, x="algorithm", y="value",
                          ax=axes[1])
prp.set_xticklabels(prp.get_xticklabels(), rotation=90)
prp.set_ylabel('Precision', size=18)
prp.set_xlabel('Algorithm (Best Conf.)', size=18)
prp.tick_params(axis='both', labelsize=16)
plt.setp(prp.artists, edgecolor = 'k', facecolor='w')
plt.setp(prp.lines, color='k')
myboxp = prp.artists[7]
# Change the appearance of that box
myboxp.set_facecolor('white')
myboxp.set_edgecolor('red')
for line in prp.get_lines()[41:47]:
   line.set_color('red')

prp.set_ylim(0.840, 0.870)



#### Recreate subfigure 2a

ridge1 = ridge[['algorithm', 'kernel', 'mean_test_balanced_accuracy', 'mean_test_recall', 'mean_test_precision', 'mean_test_f1' ]]
lasso1 = lasso[['algorithm', 'kernel', 'mean_test_balanced_accuracy', 'mean_test_recall', 'mean_test_precision', 'mean_test_f1' ]]
ela1 = ela[['algorithm', 'kernel', 'mean_test_balanced_accuracy', 'mean_test_recall', 'mean_test_precision', 'mean_test_f1' ]]
svm1 = svm[['algorithm', 'kernel', 'mean_test_balanced_accuracy', 'mean_test_recall', 'mean_test_precision', 'mean_test_f1' ]]
rf1 = rf[['algorithm', 'kernel', 'mean_test_balanced_accuracy', 'mean_test_recall', 'mean_test_precision', 'mean_test_f1' ]]
lda1 = lda[['algorithm', 'kernel', 'mean_test_balanced_accuracy', 'mean_test_recall', 'mean_test_precision', 'mean_test_f1' ]]
gbm1 = gbm[['algorithm', 'kernel', 'mean_test_balanced_accuracy', 'mean_test_recall', 'mean_test_precision', 'mean_test_f1' ]]
xgb1 = xgb[['algorithm', 'kernel', 'mean_test_balanced_accuracy', 'mean_test_recall', 'mean_test_precision', 'mean_test_f1' ]]
dtr1 = dtr[['algorithm', 'kernel', 'mean_test_balanced_accuracy', 'mean_test_recall', 'mean_test_precision', 'mean_test_f1' ]]


algo_frames = [ridge1, lasso1, ela1, rf1, svm1, gbm1, xgb1, lda1, dtr1] #add back lda, xgb
algo_frame = pd.concat(algo_frames)

algo_frame.rename(columns={'algorithm': 'Algorithm','mean_test_balanced_accuracy':'Avg. Balanced Accuracy','mean_test_precision': 'Avg. Precision'}, inplace=True)
# mean of precision and balanced accuracy to reach a compromise
algo_frame['Avg Precision/Balanced Acc.']=(algo_frame["Avg. Precision"]+algo_frame["Avg. Balanced Accuracy"])/2

sns.set_context("paper", rc={"font.size":22,"axes.titlesize":16,"axes.labelsize":22, "axes.ticksize":16})
sns.set(font_scale = 1.3)  
sns.set_style("white")
f, axes = plt.subplots(1,3)
a=sns.stripplot(x="Algorithm", y="Avg. Balanced Accuracy", data=algo_frame, ax=axes[0], alpha=0.2)
a.set_xticklabels(a.get_xticklabels(), rotation=90)
b=sns.stripplot(x="Algorithm", y="Avg. Precision", data=algo_frame, ax=axes[1],  alpha=0.2)
b.set_xticklabels(b.get_xticklabels(), rotation=90)
c=sns.stripplot(x="Algorithm", y="Avg Precision/Balanced Acc.", data=algo_frame, ax=axes[2],  alpha=0.2)
c.set_xticklabels(c.get_xticklabels(), rotation=90)
plt.tight_layout()

# Recreate Supplementary Figure S3

sns.scatterplot(data=algo_frame, x="Avg. Precision", y="Avg. Balanced Accuracy", hue="Algorithm", alpha=0.2)


# Recreate Supplementary Figure S4 in Supplementary Material

xgb.rename(columns={'mean_test_balanced_accuracy':'Avg. Balanced Accuracy',
                    'mean_test_precision': 'Avg. Precision',
                    'param_reg_alpha':'Alpha (L1)',
                    'param_reg_lambda':'Lambda (L2)',
                    'param_n_estimators':'N Estimators',
                    'param_learning_rate':'Learning Rate'},inplace=True)


xgb['Lambda (L2)'].replace(np.NaN,'No Penalty', inplace=True)
xgb['Alpha (L1)'].replace(np.NaN,'No Penalty', inplace=True)
f, axes = plt.subplots(4,2)
a=sns.boxplot(x="Alpha (L1)", y="Avg. Balanced Accuracy", data=xgb, ax=axes[3,0], color='white')#l1
a=sns.stripplot(x="Alpha (L1)", y="Avg. Balanced Accuracy", data=xgb, ax=axes[3,0], color='r', alpha=0.2)
b=sns.boxplot(x="Alpha (L1)", y="Avg. Precision", data=xgb, ax=axes[3,1], color='white')
b=sns.stripplot(x="Alpha (L1)", y="Avg. Precision", data=xgb, ax=axes[3,1],color='r', alpha=0.2)
c=sns.boxplot(x="Lambda (L2)", y="Avg. Balanced Accuracy", data=xgb, ax=axes[2,0], color='white')
c=sns.stripplot(x="Lambda (L2)", y="Avg. Balanced Accuracy", data=xgb, ax=axes[2,0], color='r', alpha=0.2) #l2
d=sns.boxplot(x="Lambda (L2)", y="Avg. Precision", data=xgb, ax=axes[2,1], color='white')
d=sns.stripplot(x="Lambda (L2)", y="Avg. Precision", data=xgb, ax=axes[2,1], color='r', alpha=0.2)
e=sns.boxplot(x="Learning Rate", y="Avg. Balanced Accuracy", data=xgb, ax=axes[1,0], color='white')
e= sns.stripplot(x="Learning Rate", y="Avg. Balanced Accuracy", data=xgb, ax=axes[1,0], color='r', alpha=0.2)#l2
f=sns.boxplot(x="Learning Rate", y="Avg. Precision", data=xgb, ax=axes[1,1], color='white')
f= sns.stripplot(x="Learning Rate", y="Avg. Precision", data=xgb, ax=axes[1,1], color='r', alpha=0.2)#l2
g=sns.boxplot(x="N Estimators", y="Avg. Balanced Accuracy", data=xgb, ax=axes[0,0], color='white')
g=sns.stripplot(x="N Estimators", y="Avg. Balanced Accuracy", data=xgb, ax=axes[0,0], color='r', alpha=0.2)
h=sns.boxplot(x="N Estimators", y="Avg. Precision", data=xgb, ax=axes[0,1], color='white') 
h=sns.stripplot(x="N Estimators", y="Avg. Precision", data=xgb, ax=axes[0,1], color='r', alpha=0.2) 
