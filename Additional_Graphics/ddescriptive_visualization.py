import pandas as pd, numpy as np, os, seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Microsoft Sans Serif"
matplotlib.rcParams['font.family'] = "sans-serif"

%matplotlib qt

df = pd.read_csv('SHR76_19.csv', delimiter=',')


# recreate Figure 1 

size=df.groupby(['Year', 'Solved']).size()
size=size.reset_index()
size.rename(columns ={list(size)[2]:'Homicides'}, inplace=True)

rest_pivot = pivot_state_share.groupby('Year')['Ratio of Solved'].agg(['min','max','mean']).reset_index().rename_axis(None, axis=1) # obtain ratio of solved
rest_pivot.reset_index(inplace=True)


matplotlib.rcParams.update({'font.size': 45})
fig, axes = plt.subplots(nrows=1, ncols=2)
first=sns.lineplot(x="Year", y="Homicides", hue="Solved", data=size, marker="o", markersize=5, ci=None, palette="mako", ax=axes[0])
first.set_ylim(0, 20000)
first.grid()
first.set_ylabel('N of Homicides', size=18)
first.tick_params(axis='both', labelsize=16)
first.set_xlabel('Year', size=18)
second=rest_pivot.plot(x='Year', y='mean', c='black', ax=axes[1])
plt.fill_between(x='Year',y1='min',y2='max', data=rest_pivot, color='grey', alpha=0.5)
second.get_legend().set_visible(False)
plt.grid()
second.set_ylabel('Ratio of Solved', size=18)
second.tick_params(axis='both', labelsize=16)
second.set_xlabel('Year', size=18)


### Recreate Figure S1 in Supplementary Material

state_init = ['AL','AK',	'AZ',	'AR',	'CA',	'CO',	'CT',	'DE',	'DC',	'FL',	
              'GA',	'HI',	'ID',	'IL',	'IN',	'IA',	'KS',	'KY',	'LA',	'ME',	
              'MD',	'MA',	'MI',	'MN',	'MS',	'MO',	'MT',	'NE',	'NV',	'NH',	
              'NJ',	'NM',	'NY',	'NC',	'ND',	'OH',	'OK',	'OR',	'PA',	'RI',	
              'SC',	'SD',	'TN',	'TX',	'UT',	'VT',	'VA',	'WA',	'WV',	'WI',	'WY', 'AL','AK',	'AZ',	'AR',	'CA',	'CO',	'CT',	'DE',	'DC',	'FL',	
              'GA',	'HI',	'ID',	'IL',	'IN',	'IA',	'KS',	'KY',	'LA',	'ME',	
              'MD',	'MA',	'MI',	'MN',	'MS',	'MO',	'MT',	'NE',	'NV',	'NH',	
              'NJ',	'NM',	'NY',	'NC',	'ND',	'OH',	'OK',	'OR',	'PA',	'RI',	
              'SC',	'SD',	'TN',	'TX',	'UT',	'VT',	'VA',	'WA',	'WV',	'WI',	'WY']

solv = df[df['Solved']=='Yes']
unso = df[df['Solved']=='No']
s_solved = solv.groupby('State').size()
s_unso = unso.groupby('State').size()
s_solved = pd.DataFrame(s_solved)
s_solved['Solved?']='Yes'
s_unso = pd.DataFrame(s_unso)
s_unso['Solved?']='No'
s_all = pd.concat([s_solved, s_unso])
s_all.reset_index(inplace=True)
s_all['StateInit']=state_init
s_all.rename(columns={list(s_all)[1]:'Homicides'}, inplace=True)
size.rename(columns ={list(size)[2]:'Homicides'}, inplace=True)
s_all.sort_values('State', inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(9, 6))
first50= sns.histplot(s_all[:50], x='StateInit', hue='Solved?', weights='Homicides', 
                  multiple='stack', ax=axes[0])
first50.set_xlabel('State', size=18)
first50.tick_params(axis='both', which='major', labelsize=14)
first50.set_ylabel('Homicides', size=18)
first50.set_ylim([0,125000])

second50 = sns.histplot(s_all[51:], x='StateInit', hue='Solved?', weights='Homicides', 
                  multiple='stack', ax=axes[1])
second50.set_xlabel('State', size=18)
second50.tick_params(axis='both', which='major', labelsize=14)
second50.set_ylabel('Homicides', size=18)
second50.set_ylim([0,125000])



# Recreate Figure S2 in Supplementary Material

mapd = df.loc[df['Source']=='MAP']

y_map  = mapd.groupby(['Year']).size()
y_map = pd.DataFrame(y_map)
y_map.reset_index(inplace=True)

years = np.array(['1990', '1991', '1992', '1993', '1994', '1995', '1995',
                  '1996', '1997', '1998', '1999', '2000',
                  '2001', '2002', '2003', '2004', '2005', '2006', '2007',
                  '2008', '2009', '2010', '2011', '2012', '2013', '2014',
                  '2015', '2016', '2017', '2018', '2019'])

years = pd.Series(years)

years = pd.DataFrame(years)

years = years.rename(columns={0: 'Year'})
years["Year"]=pd.to_numeric(years["Year"])
merge = y_map.merge(years, on='Year', how='right') # join
merge = merge.rename(columns={0: 'MAP Homicides'}) # rename column
merge['MAP Homicides'] = merge['MAP Homicides'].replace(np.nan, 0) # replace missing
merge= merge.sort_values(by='Year', ascending=True) # sort
merge = merge[merge['Year'] != 2019]
merge = merge[merge['Year'] != 1990]

ts = sns.lineplot(data=merge, x="Year", y="MAP Homicides", marker='o', color='green')
ts.set_ylabel('MAP Observations', size=18)
ts.set_xlabel('Year (beginning with First MAP Obtained Request)', size=18)
ts.tick_params(axis='both', which='major', labelsize=14)
ts.grid()
