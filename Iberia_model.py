#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:10:29 2019

@author: Daniel
"""

import pandas as pd
from sklearn import tree
import sklearn.metrics as sm
import graphviz
from plotnine import *
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from xgboost import XGBClassifier

# Change visualization preferences
pd.options.display.max_columns = 101
pd.options.display.max_rows = 200
pd.options.display.precision = 7

# Import data
con_data = pd.read_csv("V4_final_connections_db.csv", header = 0)
con_data.shape[0]
con_data.Missed.value_counts(normalize = True)
# Change the minimum departure minute to -15
#con_data.loc[con_data.delay_minutes_departure < -15, 'delay_minutes_departure'] = -15
#con_data.delay_minutes_departure.describe()

# Data summary
con_data.head()
con_data.tail()
con_data.columns
con_data.shape
con_data.Missed.value_counts(normalize=True) # Target variable distribution

# Create boolean business/leisure 
con_data["Business_leisure"] = round(con_data.num_ratio_business_trips)
con_data.Business_leisure = con_data.Business_leisure.replace(1, "Business")
con_data.Business_leisure = con_data.Business_leisure.replace(0, "Leisure")

# Select the column to use in the model
con_tree = con_data.loc[:,['airport_depart','airport_arrive','airport_arrive_2',
                           'sched_arrive','sched_depart_2',
                           'delay_minutes_departure','Flight_Region',
                           'original_transit_time','with_children',
                           'has_suitcases', 
                           'seat_row', 'seat_type', 
                           'Weekday_depart','Time_of_day_depart',
                           'Business_leisure', 'class_type',
                           'Arrive_airport_MAD','OTT_less_40',
                           'Time_of_day_depart_2','Weekday_depart_2',
                           'Month_depart', 'est_delay_arrival', 'recovery_time',
                           'Ratio_OTT']]
con_tree.head()

# Check NaN and substitute by "No_data"
con_tree.isnull().sum()
# Replace NaN by 'No_info' in categorical variables 
con_tree.seat_type.fillna(value = "No_info", inplace = True)
con_tree.Business_leisure.fillna(value = "No_info", inplace = True)
con_tree.Flight_Region.fillna(value = "No_info", inplace = True)
# Remove 'JM' in seat.row (typo) and transform to numeric
con_tree.loc[con_tree.seat_row == 'JM', 'seat_row'] = np.nan
con_tree.seat_row = pd.to_numeric(con_tree.seat_row)
# Replace NaN by the median in numerical variables 
con_tree.seat_row.fillna(value = con_tree.seat_row.median(), inplace = True) 
con_tree.delay_minutes_departure.fillna(value = con_tree.delay_minutes_departure.median(), inplace = True) #by 0 to test
con_tree.est_delay_arrival.fillna(value = con_tree.est_delay_arrival.median(), inplace = True)
con_tree.recovery_time.fillna(value = con_tree.recovery_time.median(), inplace = True)
con_tree.Ratio_OTT.fillna(value = con_tree.Ratio_OTT.median(), inplace = True)
# Re-check (all variables sholud be 0)
con_tree.isnull().sum()


# Transform factor variables to numeric variables for modelling
con_tree.info()

le_enc_list = [] # Label enconder
le_enc_names = [] # Names for the different label encoders

for i in con_tree.select_dtypes(["object"]).columns:
    le = preprocessing.LabelEncoder()
    le_enc_list.append(le)
    le_enc_names.append(i)
    # Fit labEncoder 
    le.fit(con_tree[i])
    con_tree[i] = le.transform(con_tree[i]) 


# Create the dataset with the target
target = con_data.Missed
             
# Separate in train and test (80/20%)          
X_train, X_test, y_train, y_test = train_test_split(con_tree, target, test_size=0.20, random_state=42, shuffle = True)

###############################################################################
###                          Model preparation                              ###
###############################################################################

'''
# Parameters used in the GridSearchCV
param_grid = {'booster':["gbtree"],
              'eta':[0.01,0.02,0.03,0.04],
              'gamma':[0,1,5],
              'max_depth':[3,4,5,6,7],
              'subsample':[0.8, 0.9,1],
              'colsample_bytree':[0.8,0.9,1]
              }

After that we have that our best model is:
best_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.8, eta=0.01, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=7,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=4, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=123, silent=1,
              subsample=1, verbosity=1)
'''
# Create the model with the "best model" parameters
param_grid = {'booster':["gbtree"]}
# Create the model
model_iberia_grid = GridSearchCV(estimator = XGBClassifier(base_score=0.5, 
                                                      colsample_bylevel=1,
                                                      colsample_bynode=1, 
                                                      colsample_bytree=0.8, 
                                                      eta=0.01, 
                                                      gamma=0,
                                                      learning_rate=0.1, 
                                                      max_delta_step=0, 
                                                      max_depth=7,
                                                      min_child_weight=1, 
                                                      missing=None, 
                                                      n_estimators=1000,
                                                      nthread=4, 
                                                      objective='binary:logistic', 
                                                      random_state=0,
                                                      reg_alpha=0, 
                                                      reg_lambda=1, 
                                                      scale_pos_weight=1, 
                                                      seed=123, 
                                                      silent=1,
                                                      subsample=1, 
                                                      verbosity=1), 
                     param_grid = param_grid, 
                     cv = 10) 
# Run the model
model_iberia = model_iberia_grid.fit(X_train, y_train)
# Result of crossvalidation
cvr=model_iberia.cv_results_ # To check in the console
#pd.DataFrame(cvr).to_excel("CrossValidation.xlsx") # Save the cv_results
model_iberia.best_estimator_
# Save model
#model_iberia.best_estimator_.save_model("xgb_model_v1")


# Create a plot showing the important variables
iberia_imp_iberia = pd.DataFrame({"variables":X_train.columns,"importance":model_iberia.best_estimator_.feature_importances_})
iberia_imp_iberia = iberia_imp_iberia.sort_values("importance", ascending = True)
iberia_imp_iberia
iberia_imp_iberia["features_plot"] = ['Seat type',
                 'Weekday of departure (2. flight)',
                 'Weekday of departure (1. flight)',
                 'Fare Category (Economy, Business)', 
                 'Scheduled Departure (2. flight)',
                 'Scheduled Arrival(1st flight)',
                 'Month of departure (1. flight)',
                 'Business/leisure flight',
                 'Seat row',
                 'Travel with luggage',
                 'Departure Airport (1. flight)',
                 'Travel with children',
                 'Estimated delay at arrival (1. flight)',
                 'Delay at departure in min',
                 'Arrival Airport (2. flight)',
                 'Time of the day at departure (1. flight)',
                 'Time recovered during 1. flight',
                 'Time of the day at departure (2. flight)',   
                 'Transit time',                 
                 'Flight Region',
                 'Arrival Airport (1. flight)',
                 'Ratio departure delay / Transit time',
                 'Madrid as Arrival Airport (1. flight)',
                 'Transit time less than 40 min']

# Plot important variables
iberia_imp_iberia.features_plot = pd.Categorical(iberia_imp_iberia.features_plot, iberia_imp_iberia.features_plot, ordered =True)
iberia_imp_iberia.importance = iberia_imp_iberia.importance*100
iberia_imp_iberia.to_csv("feature_importance.csv")
p1 = ggplot(aes(x ="features_plot", y= "importance"), iberia_imp_iberia) +\
 geom_bar(stat="identity") +\
 coord_flip() +\
 labs(x = "Features", 
      y = "Percentage of importance",
      title = "Feature importances") +\
 theme_minimal() +\
      theme(axis_text=element_text(size = 15),
            axis_title=element_text(size = 20),
            title = element_text(size = 25))

p1
# ggsave(p1, 'Important_variables.png', width=15, height=10, dpi=180)
# iberia_imp_iberia.to_excel('Important_variables.xlsx')

# Predict the model from test dataset
pred_iberia = model_iberia.best_estimator_.predict(X_test) # Label
predict_prob_iberia = pd.DataFrame(model_iberia.best_estimator_.predict_proba(X_test)) # Probability

###############################################################################
###                          Metrics                                        ###
###############################################################################
# Model Accuracy
model_iberia.best_score_
# Confusion matrix
sm.confusion_matrix(y_test, pred_iberia)
print(sm.classification_report(y_test, pred_iberia))


###############################################################################
###                      Population analysis                                ###
###############################################################################
# Create Dataframe with the predicted probability for each target class
predict_prob_iberia = pd.DataFrame(model_iberia.best_estimator_.predict_proba(X_test))
# Change columns name
predict_prob_iberia.columns = model_iberia.best_estimator_.classes_
# Add target variable
predict_prob_iberia["Missed"] = y_test.values

# Extract from the 'Yes'column (predicted Miss-connection) and the 'Missed' column (Real value)
missed_prob = predict_prob_iberia.loc[:, ["Yes","Missed"]]
missed_prob = missed_prob.sort_values("Yes", ascending = False) # Sort probabilities



# Prepare uplift plot to study the Percentage of missconection covered per % of population
missed_prob["Col_num"] = np.arange(1, missed_prob.shape[0]+1)
# Create the cut_100 column 
missed_prob["cut_100"] = pd.cut(missed_prob.Col_num, bins = 100, 
           labels=np.arange(1,101), 
           include_lowest=False)

missed_prob.loc[missed_prob.cut_100 == 1,:].groupby('Missed', as_index = False).count()
# Calculate the percentage of misscon per population
perc_missed = pd.crosstab(missed_prob.cut_100, 
                          missed_prob.Missed, 
                          normalize = 'index')
perc_missed.reset_index(inplace =True)
perc_missed.Yes = perc_missed.Yes*100
# Assign color column to improve the plot
perc_missed["Color"] = np.nan
perc_missed.loc[perc_missed.cut_100 == 1,"Color"] = 'Red'
perc_missed.loc[perc_missed.cut_100.isin([2,3]),"Color"] = 'Yellow' 
perc_missed.loc[~perc_missed.cut_100.isin([1,2,3]),"Color"] = 'Grey' 
#perc_missed.to_csv("perc_missed.csv")
# Plot the analysis
p2 = ggplot(aes(x = "cut_100", y = "Yes", fill = "Color"), perc_missed.iloc[:31,:]) +\
 geom_bar(aes(size = 15), stat ='identity',show_legend = False) +\
 labs(x = "Percentage of population", 
      y = "Percentage of missingh the flight",
      title = "Model Efficiency for each percentage of population") +\
      scale_fill_manual(values = ['dimgrey','firebrick','goldenrod']) +\
      scale_x_discrete(breaks = range(0,105, 5)) +\
      geom_hline(yintercept = 0.7, linetype = 'dashed', alpha = 0.5) +\
      theme_minimal() +\
      theme(axis_text=element_text(angle=90, size = 15),
            axis_title=element_text(size = 20),
            title = element_text(size = 25))
      

p2
# ggsave(p2, 'Uplift_xgboost.png', width=15, height=10, dpi=180)

# Analysis from the 1%
missed_prob_top1 = missed_prob.loc[missed_prob.cut_100 == 1,:]
missed_prob_top1.Missed.value_counts()
missed_prob_top1.Missed.value_counts(normalize = True)

# Analysis from the 2%
missed_prob_top2 = missed_prob.loc[missed_prob.cut_100 == 2,:]
missed_prob_top2.Missed.value_counts()
missed_prob_top2.Missed.value_counts(normalize = True)

# Analysis from the 3%
missed_prob_top3 = missed_prob.loc[missed_prob.cut_100 == 3,:]
missed_prob_top3.Missed.value_counts()
missed_prob_top3.Missed.value_counts(normalize = True)

# Analysis from the 2-3%
missed_prob_top2_3 = missed_prob.loc[missed_prob.cut_100.isin([2,3]),:]
missed_prob_top2_3.Missed.value_counts()
missed_prob_top2_3.Missed.value_counts(normalize = True)

'''
# Results from the model:

1% -> Yes: 253 (93.3%) // No: 18
2% -> Yes: 217 (80.4%) // No: 53
3% -> Yes: 151 (55.9%) // No: 119    

Accumated missconnection among 2-3% of the population:
    2-3% -> Yes: 368 (68.1%) // No: 172


Size of missconnection recovered in 1-3% of the population:
1% -> (254/1207)*100 = 20.96%
2% -> (217/1207)*100 = 17.98%
3% -> (151/1207)*100 = 12.51%
2-3% -> ((386)/1207)*100 = 30.49%
1-3%  -> ((254+386)/1207)*100 = 51.45%

# missed_prob.Missed.value_counts() # To check the number of missed real values
'''




