import pandas as pd
from functools import reduce
import numpy as np

# reading data in
target = pd.read_csv('cotinine.csv')
body = pd.read_csv('bodymeasures.csv')
demo = pd.read_csv('demographics.csv')
diet = pd.read_csv('dietbehavior.csv')
smoking = pd.read_csv('householdsmoking.csv')
income = pd.read_csv('income.csv')
phys = pd.read_csv('physicalactivity.csv')

# dropping participants with missing cotinine measures
target = target.dropna( how='any', subset=['LBXCOT'])
# replacing missing value codes in demographic data
demo['DMDBORN4'] = demo['DMDBORN4'].replace([77, 99], np.nan)
demo['DMDYRUSZ'] = demo['DMDYRUSZ'].replace([77, 99], np.nan)
demo['DMDEDUC2'] = demo['DMDEDUC2'].replace([7, 9], np.nan)
demo['DMDMARTZ'] = demo['DMDMARTZ'].replace([77, 99], np.nan)
# dropping participants who did not have a body measures exam and dropping feature with no data
body = body[body['BMDSTATS'] != 4]
body.drop(columns=['BMIRECUM'], inplace=True)
# replacing missing value codes in diet behavior data
diet['DBQ010'] = diet['DBQ010'].replace([7, 9], np.nan)
diet['DBD030'] = diet['DBD030'].replace([777777, 999999], np.nan)
diet['DBD030'] = diet['DBD030'].replace(666666, 1096)
diet['DBD041'] = diet['DBD041'].replace([777777, 999999], np.nan)
diet['DBD041'] = diet['DBD041'].replace(666666, 366)
diet['DBD050'] = diet['DBD050'].replace([777777, 999999], np.nan)
diet['DBD055'] = diet['DBD055'].replace([777777, 999999], np.nan)
diet['DBD055'] = diet['DBD055'].replace(666666, 731)
diet['DBD061'] = diet['DBD061'].replace([777777, 999999], np.nan)
diet['DBD061'] = diet['DBD061'].replace(666666, 1096)
diet['DBQ073A'] = diet['DBQ073A'].replace([77, 99], np.nan)
diet['DBQ700'] = diet['DBQ700'].replace([7, 9], np.nan)
diet['DBQ197'] = diet['DBQ197'].replace([7, 9], np.nan)
diet['DBQ223A'] = diet['DBQ223A'].replace([77, 99], np.nan)
diet['DBQ229'] = diet['DBQ229'].replace([7, 9], np.nan)
diet['DBQ235A'] = diet['DBQ235A'].replace([7, 9], np.nan)
diet['DBQ235B'] = diet['DBQ235B'].replace([7, 9], np.nan)
diet['DBQ235C'] = diet['DBQ235C'].replace([7, 9], np.nan)
diet['DBQ301'] = diet['DBQ301'].replace([7, 9], np.nan)
diet['DBQ330'] = diet['DBQ330'].replace([7, 9], np.nan)
diet['DBQ360'] = diet['DBQ360'].replace([7, 9], np.nan)
diet['DBQ370'] = diet['DBQ370'].replace([7, 9], np.nan)
diet['DBD381'] = diet['DBD381'].replace([7777, 9999], np.nan)
diet['DBQ390'] = diet['DBQ390'].replace([7, 9], np.nan)
diet['DBQ400'] = diet['DBQ400'].replace([7, 9], np.nan)
diet['DBD411'] = diet['DBD411'].replace([7777, 9999], np.nan)
diet['DBQ421'] = diet['DBQ421'].replace([7, 9], np.nan)
diet['DBQ424'] = diet['DBQ424'].replace([7, 9], np.nan)
diet['DBD895'] = diet['DBD895'].replace([7777, 9999], np.nan)
diet['DBD895'] = diet['DBD895'].replace([5555], 22)
diet['DBD900'] = diet['DBD900'].replace([5555], 22)
diet['DBD900'] = diet['DBD900'].replace([7777, 9999], np.nan)
diet['DBD905'] = diet['DBD905'].replace([7777, 9999], np.nan)
diet['DBD905'] = diet['DBD905'].replace([5555], 91)
diet['DBD910'] = diet['DBD910'].replace([7777, 9999], np.nan)
diet['DBD910'] = diet['DBD910'].replace([5555], 91)
diet['CBQ596'] = diet['CBQ596'].replace([7, 9], np.nan)
diet['CBQ606'] = diet['CBQ606'].replace([7, 9], np.nan)
diet['CBQ611'] = diet['CBQ611'].replace([7, 9], np.nan)
diet['DBQ930'] = diet['DBQ930'].replace([7, 9], np.nan)
diet['DBQ935'] = diet['DBQ935'].replace([7, 9], np.nan)
diet['DBQ940'] = diet['DBQ940'].replace([7, 9], np.nan)
diet['DBQ945'] = diet['DBQ945'].replace([7, 9], np.nan)
#replacing missing value codes in household smoker data
smoking['SMD460'] = smoking['SMD460'].replace([777, 999], np.nan)
smoking['SMD470'] = smoking['SMD470'].replace([777, 999], np.nan)
#replacing missing value codes in income data
income['INDFMMPC'] = income['INDFMMPC'].replace([7, 9], np.nan)
#replace missing value codes in physical activity data
# https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_PAQ.htm
phys['PAQ605'] = phys['PAQ605'].replace([7, 9], np.nan)
phys['PAQ610'] = phys['PAQ610'].replace([77, 99], np.nan)
phys['PAD615'] = phys['PAD615'].replace([7777, 9999], np.nan)
phys['PAQ620'] = phys['PAQ620'].replace([7, 9], np.nan)
phys['PAQ625'] = phys['PAQ625'].replace([77, 99], np.nan)
phys['PAD630'] = phys['PAD630'].replace([7777, 9999], np.nan)
phys['PAQ635'] = phys['PAQ635'].replace([7, 9], np.nan)
phys['PAQ640'] = phys['PAQ640'].replace([77, 99], np.nan)
phys['PAD645'] = phys['PAD645'].replace([7777, 9999], np.nan)
phys['PAQ650'] = phys['PAQ650'].replace([7, 9], np.nan)
phys['PAQ655'] = phys['PAQ655'].replace([77, 99], np.nan)
phys['PAD660'] = phys['PAD660'].replace([7777, 9999], np.nan)
phys['PAQ665'] = phys['PAQ665'].replace([7, 9], np.nan)
phys['PAQ670'] = phys['PAQ670'].replace([77, 99], np.nan)
phys['PAD675'] = phys['PAD675'].replace([7777, 9999], np.nan)
phys['PAD680'] = phys['PAD680'].replace([7777, 9999], np.nan)
# limiting to only participants for which we have cotinine measures
seqn = list(target['SEQN'])
body = body[body['SEQN'].isin(seqn)]
demo = demo[demo['SEQN'].isin(seqn)]
diet = diet[diet['SEQN'].isin(seqn)]
smoking = smoking[smoking['SEQN'].isin(seqn)]
income = income[income['SEQN'].isin(seqn)]
phys = phys[phys['SEQN'].isin(seqn)]
# merging all features together based on participant number
dfs = [body, demo, diet, smoking, income, phys, target]
all_data = reduce(lambda  left,right: pd.merge(left,right,on=['SEQN'], how='outer'), dfs)
X = all_data.drop(columns=['LBXCOT','LBDCOTLC','LBXHCOT','LBDHCOLC'])
y = all_data['LBXCOT']
# lots of missing values/nonresponse in these features, consider dropping:
# BMIWT,BMXRECUM,BMIRECUM,BMXHEAD,BMIHEAD
# PAQ610,PAD615,PAQ620
# DBQ010,DBD030,DBD041,DBD050,DBD055,DBD061,DBQ073A,DBQ073B,DBQ073C,DBQ073D,DBQ073E,DBQ073U
# split into training where ridexmon is 1 and test where ridexmon is 2. First impute missing values with 1 and 2 randomly in this column.
X_train = X[X['RIDEXMON'] == 1]
y_train = all_data[all_data['RIDEXMON'] == 1][['SEQN', 'LBXCOT']]
X_test = X[X['RIDEXMON'] != 1]
y_test = all_data[all_data['RIDEXMON'] != 1][['SEQN', 'LBXCOT']]
X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
