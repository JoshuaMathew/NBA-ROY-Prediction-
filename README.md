# NBA-ROY-Prediction-

## Intro

This project uses machine learning algorithms to predict the 2020 Rookie of the Year (ROY) for the National Basketball Association (NBA). It must be noted that a rookie is defined as a player in their first season of the NBA and we are using current data from a third of the way through the 2019-2020 season to predict the ROY for this season. Previous ROY winners were examined using the detailed statistics provided from Basketball Reference [1], a well-known website documenting an extremely wide range of statistics for each player previously or currently in the NBA. All data was scraped from Basketball Reference for each of the previous ROY winners from 2001-2019 to determine which statistics were highly correlated with winning and which were not. The correlation chart is shown below.

![Correlation](https://github.com/JoshuaMathew/NBA-ROY-Prediction-/blob/master/correlation.png)

Examining the chart, it's clear that certain statistics should be excluded from the data because they are either highly correlated with another statistic or are poorly correlated with winning the ROY. Variables least correlated with winning include three-pointers made (3P), three-pointers attempted (3PA), and three-point percentage (3P%). Total rebounds per game (TRB), rebounds per game (RPG), and offensive rebounds per game (ORB) were correlated with one another and therefore only RPG were included in the dataset for predicting ROY. Minutes played (MP) and games played (G) were also highly correlated with one another and therefore only MP was included in the dataset. We also created three variables: RPG rank, assist per game (APG) rank, and points per game (PPG) rank. These statistics were added as a way to gauge how a rookie compared against their individual class which is information that would otherwise be lost when all player data is combined for training and validation.

The recursive feature elimination algorithm (RFE) was used to eliminate predictors that were not useful to the outcome of the dataset. These variables were field goals attempted (FGA), ORB, and steals per game (STL). It is also worth noting that any rookie who averaged less than 10 PPG was removed from the dataset because no previous ROY in history has ever averaged less than 10 ppg. The final set of predictors for this dataset was determined to be: MP, field goals per game (FG), RPG, assists per game (AST), blocks per game (BLK), turnovers per game (TOV), points per game (PTS), field goal percentage (FG%), FT%, minutes per game (MPG), PPG, APG, PPG Rank, APG Rank, RPG Rank, and TRB. 

Once the most effective predictors were established, the dataset of ROY winners from 2001-2019 was randomly split into separate training and validation datasets. We then used three different methods: logistic regression (LR), k-nearest neighbor (KNN) and neural networks (NN) to fit the dataset. Each model was trained on the training data and tested on each validation dataset.

The overall model accuracy on the test data is not a great indicator of how good the model is because there is an unequal distribution of the two classes, there are many more non-ROYs in the data than ROYs. Because of this we are more concerned with the true positive rate (TPR) which is the proportion of actual ROYs that were correctly predicted to be ROY.

## Logistic Regression

The first model created was a logistic regression (LR) model. This model had an overall 95% accuracy on the test data. Below is a confusion matrix of the results.

| Predicted |
| :---:   | :-: | 
|  ROY  | Not ROY |
| 33  | 1  |
| 1 | 3 |

