#%reset

import pandas as pd
import numpy as np


read="D://Delay analysis/Data/"
write = "D://Delay analysis/Analysis/"
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import functools as ft
def SummaryDesc(Data):
    Summary = Data.describe(include = 'all')
    Summary = Summary.transpose().reset_index()
    Summary['NonMissingPercent'] = Summary['count']/len(Data.index)
    Summary = Summary.sort(['NonMissingPercent'])
    Columns = Data.columns
    Unique = pd.Series()
    for i in Columns:
        Unique[i] = Data[i].nunique()
    Unique = Unique.to_frame()
    Unique.columns = ['NUnique']
    Unique.reset_index(inplace = True)
    Dtypes = Data.dtypes
    Dtypes = Dtypes.to_frame()
    Dtypes.columns = ['DataType']
    Dtypes.reset_index(inplace = True)
    Dfs = [Summary, Unique, Dtypes]
    Summary = ft.reduce(lambda left,right: pd.merge(left,right,on='index'), Dfs)
    return Summary

data = pd.read_excel(read + 'Template_for_Variable_Buckets_Simulation_All_Data_V_2 (2)_Flags_Updated_By_Ankur_12-06-2017.xlsx')
cols = ['COV_Net_Cr',	'Avg_CrC',	'Vintage',	'growth_3m_6m',	'Age1',	'liab_calc/3mrev',	'3M-Insuff. Funds - Chq Rtn I/W',	'6M-Insuff. Funds - Chq Rtn I/W',	'12M-Insuff. Funds - Chq Rtn I/W',	'4yrMaxLateDays',	'LoanEnq6m',	'LoanEnq12m',	'CIBIL final score',	'growth_6m_12m', 'Default_Flag']

analysis_data = data[cols]
summary = SummaryDesc(analysis_data)
#analysis_data['Default_Flag'].unique()
analysis_data = analysis_data[analysis_data['Default_Flag'] != 'Rejected']
analysis_data['Default_Flag'] = analysis_data['Default_Flag'].astype(int)

"""CART analysis """
from sklearn import tree
from sklearn.tree import export_graphviz
import random
features = ['COV_Net_Cr',	'Avg_CrC',	'Vintage',	'growth_3m_6m',	'Age1',	'liab_calc/3mrev',	'3M-Insuff. Funds - Chq Rtn I/W',	'6M-Insuff. Funds - Chq Rtn I/W',	'12M-Insuff. Funds - Chq Rtn I/W',	'4yrMaxLateDays',	'LoanEnq6m',	'LoanEnq12m',	'CIBIL final score',	'growth_6m_12m']
clf = tree.DecisionTreeClassifier(min_samples_split=50, min_samples_leaf=50)
flag_0 = analysis_data[analysis_data['Default_Flag'] == 0] #3692 -> 3692*80% = 
rows = random.sample(flag_0.index, 2954)
train_80_0 = flag_0.ix[rows]
test_20_0 = flag_0.drop(rows)
flag_1 = analysis_data[analysis_data['Default_Flag'] == 1] #855
rows = random.sample(flag_1.index, 684)
train_80_1 = flag_1.ix[rows]
test_20_1 = flag_1.drop(rows)

train_80 = train_80_0.append(train_80_1)
test_20 = test_20_0.append(test_20_1)

train_80['Default_Flag'].mean()
test_20['Default_Flag'].mean()

clf = clf.fit(train_80[features], train_80['Default_Flag'])

with open(write + 'tree.dot', 'w') as f:
        export_graphviz(clf, out_file=f, feature_names = features)

train_80['Flag'] = np.where((train_80['CIBIL final score'] <= 620) & (train_80['COV_Net_Cr'] > 80), 1, 0)
train_80.groupby(['Flag']).agg({'Default_Flag':np.mean, 'CIBIL final score': np.size})
test_20['Flag'] = np.where((test_20['CIBIL final score'] <= 620) & (test_20['COV_Net_Cr'] > 80), 1, 0)
test_20.groupby(['Flag']).agg({'Default_Flag':np.mean, 'CIBIL final score': np.size})

"""Looking only flagged population """
summary_data = SummaryDesc(data)
data['Flag'] = np.where((data['CIBIL final score'] <= 620) & (data['COV_Net_Cr'] > 80), 1, 0)
flag_pop = data[data['Flag'] == 1]
flag_pop = flag_pop[cols + ['Flag', 'Contract ID', 'CLSAppID']]
flag_pop['Default_Flag'].value_counts()

lpm_flags = pd.read_excel(read + 'LAIDs_Risk_Flag_Tarun.xlsx')
flag_pop = pd.merge(flag_pop, lpm_flags, on = 'Contract ID', how = 'left')
flag_pop = flag_pop.rename(columns = {'Data': 'sample'})
flag_pop['sample'].value_counts(dropna = False)
flag_pop['count'] = flag_pop.groupby(['Contract ID'])['Contract ID'].transform('count')
issue = flag_pop[flag_pop['count'] > 1]
flag_pop = flag_pop.drop_duplicates()


bot_score = pd.read_excel(read + 'Score, classification from May to Jan cases - Final sheet.xlsx')
col_list = bot_score.columns
col_list = [i.encode('ascii', 'ignore') for i in col_list]
col_list = col_list[~col_list.isin('Contract ID')]
flag_pop = pd.merge(flag_pop, bot_score, on = 'CLSAppID', how = 'left')
flag_pop.drop_duplicates(inplace = True)
flag_pop['Final classification'].value_counts(dropna = False)

lpm_issue = flag_pop[flag_pop['Risk_Flag'] == 'Low Risk']

"""Creating deciles """
del data['TVA/rev']
del data['LECCCMAXB']
del data['CIBIL1']

data.dropna(inplace = True)

def decile(data, cols):
    for i in cols:
        data['dec' + '_' + i] = pd.qcut(data[i] + jitter(data[i]), 10, labels = False)
    return data

def jitter(a_series, noise_reduction=1000000):
    return (np.random.random(len(a_series))*a_series.std()/noise_reduction)-(a_series.std()/(2*noise_reduction))


cols = ['cov_cc',	'cov_cd',	'COV_Min_Bal',	'COV_Net_Cr',	'COV_CB',	'COV_OB',	'COV_CrDrR',	'COV_DbC',	'COV_CrC',	'COV_AvgBal',	'COV_CW',	'Avg_CrC',	'Vintage',	'BACr',	'CrDrR',	'DelayMonthsCount',	'CIBIL final score',	'LoanEligible',	'MonthsLastDelinquency',	'LoanInquiries',	'Cumulative MAB Charges Count',	'MaxLateDays',	'OverdueLoanCount',	'OutstandingLoanCount',	'SettledLoanCount',	'WrittenOffLoanCount',	'Number of Months of Bank Statement',	'revenue3m',	'growth_3m_6m',	'growth_6m_12m',	'liab_calc/6mrev',	'cashdep_6mrev',	'AO/outstanding',	'CL/revenue3m',	'cashdep_rev',	'liab_calc/3mrev',	'Current Liability',	'Other Term Liability',	'Average Cheque Returns',	'3MAvgChqRtn',	'6MAvgChqRtn',	'12MAvgChqRtn',	'Total Cheque Returns Inward',	'Total Cheque Returns Outward',	'LECCCMINA',	'LECCCMINB',	'LECCCMAXA',	'MaxLEMTA',	'MaxLEMTB',	'MaxLEMSA',	'MaxLEMSB',	'MaxLETA',	'MaxLETB']
data_decile = decile(data, cols)

"""Univariate analysis """
def univariate(data, cols):
    df = pd.DataFrame()
    for i in cols:
        df[i+'_pd'] = data.groupby(['dec_' + i]).agg({'Default_Flag': np.mean})
    return df

df = univariate(data_decile, cols)
df.to_excel(read + 'univariate.xlsx')
#data.to_csv(read + 'data.csv')

"""Testing github """

"""Testing github 2"""