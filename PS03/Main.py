
#%%

import pandas as pd
import ModuleReg

#%%

d= pd.read_excel('HW3.xlsx')
d.columns= ['SampleNo', 'WeeklyTrips', 'VehicleOwnership', 'AnnualIncome', 'FamilyWorkingDays', 'AvgAge']
d= d.drop('SampleNo', axis= 1)
d= d.sample(frac=1, random_state=42)

#%%

train_size= 0.7
d_train= d.head(int(len(d)* train_size))
d_test= d.head(-int(len(d)* train_size))

#%%

low_limit = 0.2
up_limit = 0.6
data= d
features= ['VehicleOwnership', 'AnnualIncome', 'FamilyWorkingDays', 'AvgAge']
target= 'WeeklyTrips'
combos = ModuleReg.get_valid_feature_combinations(data, features, target, low_limit, up_limit)

#%%

model_results = ModuleReg.train_and_evaluate_models(
            d_train, 
            d_test, 
            target, 
            combos)

#%%

ModuleReg.plot_mse_bar_chart(model_results)
ModuleReg.plot_r2adjusted_bar_chart(model_results)

#%%


