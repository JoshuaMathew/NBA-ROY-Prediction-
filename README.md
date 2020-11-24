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



|  | Predicted ROY  | Predicted Not ROY |
| :---:  | :-: | :-:|
| Actual ROY| 33  | 1  |
| Actual Not ROY| 1 | 3 |

The model correctly predicted 33 players that were not ROY, 3 players correctly that were ROY, 1 player was predicted incorrectly to be ROY, and 1 player was not predicted to be ROY that actually was. The true positive rate (TPR) was ¾ or 75% and the false positive rate (FPR) was 1/34 or 2.9%. Below is a table displaying the ROY’s that the LR model predicted for the 1990-2000 season compared to the actual ROY winners.

| Year | Actual ROY  | Predicted ROY |
| :---:  | :-: | :-:|
| 1990| David Robinson  | David Robinson  |
| 1991| Derrick Coleman | Lionel Simmons |
| 1992| Larry Johnson | Larry Johnson |
| 1993| Shaquille O'Neal | Shaquille O'Neal |
| 1994| Chris Webber | Chris Webber |
| 1995| Grant Hill | Grant Hill |
| 1996| Damon Stoudamire | Damon Stoudamire |
| 1997| Allen Iverson | Antoine Walker |
| 1998| Tim Duncan | Tim Duncan |
| 1999| Vince Carter | Vince Carter |
| 2000| Elton Brand | Elton Brand |

The LR model correctly predicted the ROY winner during these 11 seasons all but two times for a total of 81.8% accuracy. The figure below displays the weights the LR model assigned to each predictor. PPG Rank was the most influencing variable towards winning followed by APG Rank. Surprisingly, rebounding was negatively correlated with winning. This may be because rebounding and assists are negatively correlated.

![LR_weights](https://github.com/JoshuaMathew/NBA-ROY-Prediction-/blob/master/LR_weights.JPG)

The LR model was then fed the data for the current 2019-2020 rookies in the NBA and the results conclude that Ja Morant has the greatest chance to win ROY. The results can be seen below.

![LR_pred](https://github.com/JoshuaMathew/NBA-ROY-Prediction-/blob/master/LR_prob.JPG)
