# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:40:41 2019

@author: Joshua Mathew
"""
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

#%%%%%%%%%%%%%%%%%%%%%%    OBTAINING DATA
##Getting current 2020 rookie stats
# URL page to scrape
url = "https://www.basketball-reference.com/leagues/NBA_2020_rookies.html"
#this is the HTML from the given URL
html = urlopen(url)
soup = BeautifulSoup(html)

# use findALL() to get the column headers
soup.findAll('tr', limit=3)
# use getText()to extract the text we need into a list
headers = [th.getText() for th in soup.findAll('tr', limit=3)[1].findAll('th')]
# exclude the first column as we will not need the ranking order from Basketball Reference for the analysis
headers = headers[1:]


# avoid the first header row
rows = soup.findAll('tr')[1:]
player_stats = [[td.getText() for td in rows[i].findAll('td')]
            for i in range(len(rows))]

#remove empty row
player_stats = player_stats[1:]
stats20 = pd.DataFrame(player_stats, columns = headers)
#remove unnecessary columns
stats20 = stats20.drop(["Debut","Yrs","PF","Age","FT","FTA"],axis=1)

stats20.columns = ['Player', 'G', 'MP', 'FG', 'FGA', '3P', '3PA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'FG%',
       '3P%', 'FT%', 'MPG', 'PPG', 'RPG', 'APG']

#convert data to float values
cols = stats20.columns[1:]
stats20[cols] = stats20[cols].apply(pd.to_numeric, errors='coerce', axis=1)


# Getting all rookie stats from start year to 2019
startYear = 1990
startYearTrain = 2001

years = list(range(startYear,2020))

allStats= {}

for i in years:
    url = "https://www.basketball-reference.com/leagues/NBA_{}_rookies-season-stats.html".format(i)
    html = urlopen(url)
    soup = BeautifulSoup(html)
    headers = [th.getText() for th in soup.findAll('tr', limit=3)[1].findAll('th')]
    # exclude the first column as we will not need the ranking order from Basketball Reference for the analysis
    headers = headers[1:]
    
    # avoid the first header row
    rows = soup.findAll('tr')[1:]
    player_stats = [[td.getText() for td in rows[j].findAll('td')]
                for j in range(len(rows))]

    #remove empty row
    player_stats = player_stats[1:]
    stats = pd.DataFrame(player_stats, columns = headers)
    #remove unnecessary columns
    stats = stats.drop(["Debut","Yrs","PF","Age","FT","FTA"],axis=1)
    
  
    stats.columns = ['Player',  'G', 'MP', 'FG', 'FGA', '3P', '3PA',
            'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'FG%',
           '3P%', 'FT%', 'MPG', 'PPG', 'RPG', 'APG']
    
    #convert data to float values
    cols = stats.columns[1:]
    stats[cols] = stats[cols].apply(pd.to_numeric, errors='coerce', axis=1)
    allStats[i] = stats

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DATA PROCESSING
# Add PPG, APG, and RPG rankings
for i in allStats:
    allStats[i][['PPG Rank','APG Rank','RPG Rank']] = allStats[i][['PPG','APG','RPG']].rank(ascending=False)

stats20[['PPG Rank','APG Rank','RPG Rank']] = stats20[['PPG','APG','RPG']].rank(ascending=False)
# Remove players averaging less than 10ppg
for i in allStats:
    allStats[i] = allStats[i][allStats[i].PPG>10]
    index = list(range(len(allStats[i])))
    allStats[i].index = index
    
stats20 = stats20[stats20.PPG>10]

#  Create binary column for ROY. 1 for winner 0 for not
# Initialize all players to 0
for i in allStats:
    allStats[i]['Winner'] = 0
    
#  Set ROY for each year
ROYs = ['David Robinson*',
		'Derrick Coleman',
		'Larry Johnson'	,
		"Shaquille O'Neal*",
		'Chris Webber',
		'Grant Hill*' ,
		'Damon Stoudamire',
		'Allen Iverson*'	,
		'Tim Duncan',
		'Vince Carter',	
        'Elton Brand',   
        'Mike Miller',
        'Pau Gasol',
        "Amar'e Stoudemire",
        'LeBron James',
            'Emeka Okafor',
            'Chris Paul',
            'Brandon Roy',
            'Kevin Durant',
            'Derrick Rose',
            'Tyreke Evans',
            'Blake Griffin',
            'Kyrie Irving',
            'Damian Lillard',
            'Michael Carter-Williams',
            'Andrew Wiggins',
            'Karl-Anthony Towns',
            'Malcolm Brogdon',
            'Ben Simmons',
            'Luka Dončić']
    
  
# assign each ROY a 1 for the Winner column
for i in allStats:
    allStats[i].loc[allStats[i]['Player'] == ROYs[i-startYear],'Winner'] = 1

# remove all data before training year so it is not trained and can be used for validation
allStats2 = allStats.copy()
allStats_new = {}

years = list(range(startYearTrain,2020))
for i in years:
    allStats_new[i] = allStats[i]
    
allStats = allStats_new


# Combine all seasons into 1 dataframe
#seasons = list(range(len(allStats)))
seasons = list(range(startYearTrain,2020))
index = list(range(startYearTrain,2020))
for i in allStats:
    allStats[i].fillna(allStats[i].mean(),inplace = True)
    seasons[i-startYearTrain] = allStats[i]
    
combinedStats = pd.concat(seasons)

#reindex combined season stats to be in order
index = list(range(len(combinedStats))) 
combinedStats.index = index
    
# Make Correlation Matrix   
corr = combinedStats[['G', 'MP', 'FG', 'FGA', '3P', '3PA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'FG%',
       '3P%', 'FT%', 'MPG', 'PPG', 'RPG', 'APG','PPG Rank','APG Rank','RPG Rank','Winner']].corr()
                      
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
fig2 = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.savefig("heatmap.png")       

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Logistic REGRESSION
rookies = combinedStats

rookies20 = stats20

#re index rookie stats to be in numerical order
index = list(range(len(rookies20)))
rookies20.index = index

# Choosing features for model
features_options = ['G', 'MP', 'FG', 'FGA', '3P', '3PA', 'ORB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS', 'FG%',
       '3P%', 'FT%', 'MPG', 'PPG', 'RPG', 'APG','PPG Rank','APG Rank','RPG Rank']

features = [ 'MP', 'FG','RPG', 'AST',  'BLK', 'TOV', 'PTS', 'FG%',
        'FT%', 'MPG', 'PPG', 'APG','PPG Rank','APG Rank','RPG Rank','TRB']

#per game features only
#features = [ 'FG','RPG', 'FG%',
#        'FT%', 'MPG', 'PPG', 'APG','PPG Rank','APG Rank','RPG Rank']

# split data into test and training data

y = rookies['Winner']
x = rookies[features]

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.30)

x_train.index = list(range(len(x_train))) 
y_train.index = list(range(len(y_train))) 
x_test.index = list(range(len(x_test))) 
y_test.index = list(range(len(y_test))) 

# Replace NaN values with column mean
x_train.fillna(x_train.mean(),inplace = True)

# Recursive Feature Elimination
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(x_train, y_train)
print(rfe.support_)
print(rfe.ranking_)

# creating linear regression model
logit_model=sm.Logit(y_train,x_train)
result=logit_model.fit()
print(result.summary2())


# test linreg model
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

# create confusion matrix
confusionMatrix_Linreg = confusion_matrix(y_test, y_pred)
print(confusionMatrix_Linreg)

# Take test data of the selected features to make predictions
allStatsTest = allStats2.copy()
for i in allStats2:
    allStatsTest[i] = allStatsTest[i][features]

pred_players = list(range(len(allStats2)))


#%%  
# Use linreg model to predict roy for each class
for i in allStatsTest:
    pred_prob = logreg.predict_proba(allStatsTest[i])[:,1]
    pred_player_index = np.where(pred_prob == max(pred_prob))
    pred_player_indexcon = pd.to_numeric(pred_player_index[0], errors='coerce')
    pred_player = allStats2[i].iloc[pred_player_indexcon[0]].Player
    pred_players[i-startYear] = pred_player

# calculate how many ROYS were predicted correctly from 1990 to 2000
count = 0
for i in list(range(startYearTrain-startYear)):
    if ROYs[i] == pred_players[i]:
        count+= 1
accuracy = count / (startYearTrain-startYear)
print('The linear regression model correctly predicted ' + str(round(accuracy,2)) + ' of the ROYs')

#create table comparing actual vs predicted ROY for each year
Comp = pd.DataFrame(np.stack((ROYs,pred_players),axis=-1))
Comp.columns = ['Actual','Predicted']
index = list(range(startYear,2020))
Comp.index = index

# Predict 2020 ROY
LRpred20_prob = logreg.predict_proba(rookies20[features])[:,1]
pred_20_index = np.where(LRpred20_prob == max(LRpred20_prob))
pred_player20_indexcon = pd.to_numeric(pred_20_index[0], errors='coerce')
LRpred_player20 = rookies20.iloc[pred_player20_indexcon[0]].Player

print(LRpred_player20 + ' is the predicted 2020 ROY')

#%%
# Graphing weights
LR_weights = logreg.coef_[0]
LR_weights[[12,13]] = LR_weights[[12,13]]*-1

objects = features
y_pos = np.arange(len(objects))

plt.bar(y_pos, LR_weights, align='center', alpha=0.5,width = .3)
plt.xticks(y_pos, objects,rotation = 'vertical')
#plt.ylabel('Usage')
plt.title('LR Model Feature Weights')
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    KNN
#  Generate test and training data
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.30)

#normalize training data
scaler = StandardScaler()
scaler.fit(x_train)

x_train_norm = scaler.transform(x_train)
x_test_norm = scaler.transform(x_test)
rookies20_norm = scaler.transform(rookies20[features])

# Use PCA to reduce dimensions
pca = PCA(.95)
pca.fit(x_train_norm)

# PCA reduced  features from 16 to 9 dimensions

x_train_reduced = pca.transform(x_train_norm)
x_test_reduced = pca.transform(x_test_norm)
rookies20_reduced = pca.transform(rookies20_norm)

#%% KNN
# For loop to find best neighbors value
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_reduced, y_train)
    pred_i = knn.predict(x_test_reduced)
    error.append(np.mean(pred_i != y_test))

#Plotting error vs number of neighbors
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


#Create KNN model
KNclassifier = KNeighborsClassifier(n_neighbors=6)
KNclassifier.fit(x_train_reduced, y_train)

y_pred2 = pd.Series(KNclassifier.predict(x_test_reduced))

#Make confusion matrix
confusionMatrix_KNN = confusion_matrix(y_test, y_pred2)
print(confusionMatrix_KNN)
print('Accuracy of KNN classifier on test set: {:.2f}'.format(KNclassifier.score(x_test_reduced, y_test)))


#Predict 2020 ROY
KNpred20_prob = KNclassifier.predict_proba(rookies20_reduced)[:,1]
KNpred_20_index = np.where(KNpred20_prob == max(KNpred20_prob))
KNpred_player20_indexcon = pd.to_numeric(KNpred_20_index[0], errors='coerce')
KNpred_player20 = rookies20.iloc[KNpred_player20_indexcon[0]].Player

print(KNpred_player20 + ' is the predicted 2020 ROY by KNN')


#Predict past ROYs
KNpred_players = list(range(len(allStats2)))

for i in allStatsTest:
    allStats_norm = scaler.transform(allStatsTest[i])
    allStats_reduced = pca.transform(allStats_norm)
    KNpred_prob = KNclassifier.predict_proba(allStats_reduced)[:,1]
    KNpred_player_index = np.where(KNpred_prob == max(KNpred_prob))
    KNpred_player_indexcon = pd.to_numeric(KNpred_player_index[0], errors='coerce')
    KNpred_player = allStats2[i].iloc[KNpred_player_indexcon[0]].Player
    KNpred_players[i-startYear] = KNpred_player


#calculate correct predictons for 1990-2000
count = 0
for i in list(range(startYearTrain-startYear)):
    if ROYs[i] == KNpred_players[i]:
        count+= 1
accuracy_KNN = count / (startYearTrain-startYear)
print('The KNN model correctly predicted ' + str(round(accuracy_KNN,2)) + ' of the ROYs')

KNComp = pd.DataFrame(np.stack((ROYs,KNpred_players),axis=-1))
KNComp.columns = ['Actual','Predicted']
index = list(range(startYear,2020))
KNComp.index = index

#%%%%%%%%%%%%%%%%%%%%                       NEURAL NETWORK
#Initializing Neural Network
NNclassifier = Sequential()

# Adding the input layer and the first hidden layer
NNclassifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = len(features)))
# Adding the second hidden layer
NNclassifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
NNclassifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling Neural Network
NNclassifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Generate test and training data
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.30)


#normalize training data
scaler = StandardScaler()
scaler.fit(x_train)

x_train_norm = scaler.transform(x_train)
x_test_norm = scaler.transform(x_test)
rookies20_norm = scaler.transform(rookies20[features])

# Plug normalized training data into nn
NNclassifier.fit(x_train_norm, y_train, batch_size = 10, nb_epoch = 250)

# Predicting the Test set results
y_prob = NNclassifier.predict(x_test_norm)
y_pred = (y_prob > 0.5)

confusionMatrix_NN = confusion_matrix(y_test, y_pred)
print(confusionMatrix_NN)

#Predict 2020 ROY
NNpred20_prob = NNclassifier.predict(rookies20_norm)
NNpred_20_index = np.where(NNpred20_prob == max(NNpred20_prob))
NNpred_player20_indexcon = pd.to_numeric(NNpred_20_index[0], errors='coerce')
NNpred_player20 = rookies20.iloc[NNpred_player20_indexcon[0]].Player

print(NNpred_player20 + ' is the predicted 2020 ROY by the NN')

#Predict past ROYs
NNpred_players = list(range(len(allStats2)))

for i in allStatsTest:
    allStats_norm = scaler.transform(allStatsTest[i])
    NNpred_prob = NNclassifier.predict_proba(allStats_norm)
    NNpred_player_index = np.where(NNpred_prob == max(NNpred_prob))
    NNpred_player_indexcon = pd.to_numeric(NNpred_player_index[0], errors='coerce')
    NNpred_player = allStats2[i].iloc[NNpred_player_indexcon[0]].Player
    NNpred_players[i-startYear] = NNpred_player

#count how many correct ROYs were predected for 1990-2000 by NN
count = 0
for i in list(range(len(ROYs)-(startYearTrain-startYear-1))):
    if ROYs[i] == NNpred_players[i]:
        count+= 1
NN_accuracy = count / (len(ROYs)-(startYearTrain-startYear-1))
print('The neural network correctly predicted ' + str(round(NN_accuracy,2)) + ' of the ROYs')

NNComp = pd.DataFrame(np.stack((ROYs,NNpred_players),axis=-1))
NNComp.columns = ['Actual','Predicted']
index = list(range(startYear,2020))
NNComp.index = index

#%% Graphing Weights

NN_weights   = NNclassifier.get_weights()
N_weights = NN_weights[0][:,5]*-1
N_weights[[12,13,14]] =N_weights[[12,13,14]]*-1

objects = features
y_pos = np.arange(len(objects))

plt.bar(y_pos, N_weights, align='center', alpha=0.5,width = .3)
plt.xticks(y_pos, objects,rotation = 'vertical')
#plt.ylabel('Usage')
plt.title('NN Model Feature Weights')
plt.show()

# 2020 probability Bar Chart LR
plt.bar(np.arange(2),[.83,.19] , width=.8, bottom=None, align='center', data=None)
plt.xticks(np.arange(2), ['Ja Morant','Kendrick Nunn'])
plt.ylabel('Probability')
plt.xlabel('Player')
plt.text(-.1,.85,'.83')
plt.text(.9,.2,'.19')  
