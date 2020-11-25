# NBA-ROY-Prediction-

## Intro

### Feature Engineering
This project uses machine learning algorithms to predict the 2020 Rookie of the Year (ROY) for the National Basketball Association (NBA). It must be noted that a rookie is defined as a player in their first season of the NBA and we are using current data from a third of the way through the 2019-2020 season to predict the ROY for this season. Previous ROY winners were examined using the detailed statistics provided from Basketball Reference [1], a well-known website documenting an extremely wide range of statistics for each player previously or currently in the NBA. All data was scraped from Basketball Reference for each of the previous ROY winners from 2001-2019 to determine which statistics were highly correlated with winning and which were not. The correlation chart is shown below.

![Correlation](https://github.com/JoshuaMathew/NBA-ROY-Prediction-/blob/master/images/correlation.png)

Examining the chart, it's clear that certain statistics should be excluded from the data because they are either highly correlated with another statistic or are poorly correlated with winning the ROY. Variables least correlated with winning include three-pointers made (3P), three-pointers attempted (3PA), and three-point percentage (3P%). Total rebounds per game (TRB), rebounds per game (RPG), and offensive rebounds per game (ORB) were correlated with one another and therefore only RPG were included in the dataset for predicting ROY. Minutes played (MP) and games played (G) were also highly correlated with one another and therefore only MP was included in the dataset. We also created three variables: RPG rank, assist per game (APG) rank, and points per game (PPG) rank. These statistics were added as a way to gauge how a rookie compared against their individual class which is information that would otherwise be lost when all player data is combined for training and validation.

The recursive feature elimination algorithm (RFE) was used to eliminate predictors that were not useful to the outcome of the dataset. These variables were field goals attempted (FGA), ORB, and steals per game (STL). It is also worth noting that any rookie who averaged less than 10 PPG was removed from the dataset because no previous ROY in history has ever averaged less than 10 ppg. The final set of predictors for this dataset was determined to be: MP, field goals per game (FG), RPG, assists per game (AST), blocks per game (BLK), turnovers per game (TOV), points per game (PTS), field goal percentage (FG%), FT%, minutes per game (MPG), PPG, APG, PPG Rank, APG Rank, RPG Rank, and TRB. 

### Training, Validation and Test Sets
Once the most effective predictors were established, the dataset of ROY winners from 2001-2019 was randomly split into separate training and validation datasets. We then used three different methods: logistic regression (LR), k-nearest neighbor (KNN) and neural networks (NN) to fit the dataset. Each model was trained on the training data and tested on each validation dataset.

Selecting the correct ROY’s from a large group of players containing multiple ROY’s from many different NBA seasons is difficult. A more realistic test of each model is feeding several seasons, that were not used for training, into each model and selecting the rookie with the highest assigned probability as the predicted ROY for each season. A test dataset for each rookie class from 1990 to 2000 was fed into the models and the ROYs for those years were predicted using this method. The accuracy of these predictions is a better indicator of how well the models performed. 

Note that the LR and KNN models do not see any data from the validation set at all during training. The NN model is indirectly influenced by the validation data through the validation loss. However, the test data set from 1990-2000 does not influence any of the models during training.

### Model Evaluation
The overall model accuracy on the test data is not a great indicator of how good the model is because there is an unequal distribution of the two classes, there are many more non-ROYs in the data than ROYs. Because of this we are more concerned with the true positive rate (TPR) which is the proportion of actual ROYs that were correctly predicted to be ROY.

## Logistic Regression Model

The first model created was a logistic regression (LR) model. This model had an overall 95% accuracy on the test data. Below is a confusion matrix of the results on the validation set.


|  | Predicted ROY  | Predicted Not ROY |
| :---:  | :-: | :-:|
| Actual ROY| 33  | 1  |
| Actual Not ROY| 1 | 3 |

The model correctly predicted 33 players that were not ROY, 3 players correctly that were ROY, 1 player was predicted incorrectly to be ROY, and 1 player was not predicted to be ROY that actually was. The true positive rate (TPR) was ¾ or 75% and the false positive rate (FPR) was 1/34 or 2.9%. 

Below is a table displaying the ROY’s that the LR model predicted for the seperate 1990-2000 test set compared to the actual ROY winners.

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

The LR model correctly predicted the ROY winner during these 11 seasons all but two times for a total of 81.8% accuracy. 

The figure below displays the weights the LR model assigned to each predictor. PPG Rank was the most influencing variable towards winning followed by APG Rank. Surprisingly, rebounding was negatively correlated with winning. This may be because rebounding and assists are negatively correlated.

![LR_weights](https://github.com/JoshuaMathew/NBA-ROY-Prediction-/blob/master/images/LR_weight.JPG)

The LR model was then fed the data for the current 2019-2020 rookies in the NBA and the results conclude that Ja Morant has the greatest chance to win ROY. The results can be seen below.

![LR_pred](https://github.com/JoshuaMathew/NBA-ROY-Prediction-/blob/master/images/LR_prob.JPG)

## K-Nearest Neighbors Model

The second model used was the k-nearest neighbors algorithm. The data was first normalized, and the number of features was reduced from 16 to 9 using principal component analysis (PCA) to retain 95% variance before training and testing the model. 

The model was tested using k values ranging from 1 to 40. A k value of 5 resulted in the model with the lowest mean error. This model had an overall accuracy of 89% on the test data. Below is a plot of the model error rate vs the number of neighbors used.

![KNN_error](https://github.com/JoshuaMathew/NBA-ROY-Prediction-/blob/master/images/KNN_error.JPG)

Below is the confusion matrix of the results of the KNN model on the validation dataset.

|  | Predicted ROY  | Predicted Not ROY |
| :---:  | :-: | :-:|
| Actual ROY| 33  | 1  |
| Actual Not ROY| 2 | 2 |

The model correctly predicted 33 players that were not ROY, 2 players correctly that were ROY, 1 player was predicted incorrectly to be ROY, and 2 players not to be ROY that actually were. The true positive rate (TPR) was 50% and the false positive rate (FPR) was 2.9%.

The KNN model correctly predicted the ROY from 1990-2000 72.7% of the time only missing 3 out of 11. This is slightly worse than the LR method, but when given the current 2020 player data, the KNN model also predicted Ja Morant to win the ROY.

| Year | Actual ROY  | Predicted ROY |
| :---:  | :-: | :-:|
| 1990| David Robinson  | Tim Hardaway  |
| 1991| Derrick Coleman | Derrick Coleman |
| 1992| Larry Johnson | Larry Johnson |
| 1993| Shaquille O'Neal | Christian Laettner |
| 1994| Chris Webber | Anfernee Hardaway |
| 1995| Grant Hill | Grant Hill |
| 1996| Damon Stoudamire | Damon Stoudamire |
| 1997| Allen Iverson | Antoine Walker |
| 1998| Tim Duncan | Tim Duncan |
| 1999| Vince Carter | Vince Carter |
| 2000| Elton Brand | Elton Brand |

## Neural Network Model

![NN Structure](https://github.com/JoshuaMathew/NBA-ROY-Prediction-/blob/master/images/NN.JPG)

The final model used was a neural network (NN). The input layers takes in the 16 features, followed by 2 dense fully connected layers containing 6 nodes each. The ReLu activation function was used for the hidden layers. Because the output is binary, the sigmoid function was used for the output layer. The structure of the NN is shown above.

The data was again normalized before training and testing. The neural network was trained using a batch size of 10 over 250 epochs. The cross-entropy loss function and Adam optimization algorithm were employed. The validation loss curve is shown below.

![NN Loss](https://github.com/JoshuaMathew/NBA-ROY-Prediction-/blob/master/images/NN_loss.JPG)

Below is the confusion matrix for the results of the NN on the validation dataset.

|  | Predicted ROY  | Predicted Not ROY |
| :---:  | :-: | :-:|
| Actual ROY| 33  | 2  |
| Actual Not ROY| 1 | 3 |

the model correctly predicted 32 players that were not ROY, 3 players correctly that were ROY, 2 players were predicted incorrectly to be ROY, and 1 player not to be ROY that actually was. The true positive rate (TPR) was 75% and the false positive rate (FPR) was 5.9%. The model achieved an overall accuracy of 97% on the validation set.

Below are the results on the test set.

| Year | Actual ROY  | Predicted ROY |
| :---:  | :-: | :-:|
| 1990| David Robinson  | David Robinson  |
| 1991| Derrick Coleman | Derrick Coleman |
| 1992| Larry Johnson | Larry Johnson |
| 1993| Shaquille O'Neal | Christian Laettner |
| 1994| Chris Webber | Anfernee Hardaway |
| 1995| Grant Hill | Grant Hill |
| 1996| Damon Stoudamire | Damon Stoudamire |
| 1997| Allen Iverson | Shareef Abdur-Rahim |
| 1998| Tim Duncan | Tim Duncan |
| 1999| Vince Carter | Vince Carter |
| 2000| Elton Brand | Elton Brand |

The neural network model correctly predicted the ROY winner 10/11 seasons for a total of 90.9% accuracy. 

The figure below displays the predictor weights of the model. Stats associated with assists were the biggest predictors. Surprisingly turnovers were positively associated with winning even though turning the ball over is bad. This may be because turnovers are positively correlated with points and assists.

![NN Weights](https://github.com/JoshuaMathew/NBA-ROY-Prediction-/blob/master/images/NN_weights.JPG)

After inputting the 2019-2020 rookie data into the model Ja Morant was again predicted to be the 2020 ROY.

## Conclusion

All the models predicted Ja Morant to be the 2020 NBA Rookie of the Year. This is a reasonable prediction as Ja Morant is currently the consensus best performing rookie in the NBA. All models show a wide gap in winning probability between Morant and any other rookie. 

|  | NN  | LR | KNN |
| :---:  | :-: | :-:| :-:|
| Validation Set TPR (2000-20| 75%  | 75%  | 50% |
| Validation Set Accuracy| 97% | 95% | 89% |
| Test Set Proportion of  Predicted ROYs Correct (1990-2000)| 91% | 82% | 73%|

Overall the neural network was the best performing model. It was tied with the linear regression model for highest TPR, correctly predicted the most ROY’s from 1990 to 2000, and had the highest overall accuracy on the validation set. 

The linear regression was the second best performing model followed by k-nearest neighbors. All models performed significantly better than guessing. For example, if only looking at players who have a reasonable chance of winning (>10ppg) there are generally between 3-10 potential winners. Guessing will result in a 10-33% chance of selecting the correct ROY. All the models were much more accurate than this.

## References
[1]	Basketball Statistics and History,” Basketball Reference. [Online]. Available: https://www.basketball-reference.com/. [Accessed: 27-Nov-2019].W.-K. Chen, Linear Networks and Systems. Belmont, CA, USA: Wadsworth, 1993, pp. 123–135.

[2]	D. Bratulić “Predicting 2018–19 NBA's Most Valuable Player using Machine Learning,” Medium, 15-May-2019. [Online]. Available: https://towardsdatascience.com/predicting-2018-19-nbas-most-valuable-player-using-machine-learning-512e577032e3. [Accessed: 17-Dec-2019].

[3]	S. Kannan, “Predicting NBA Rookie Stats with Machine Learning,” Medium, 30-Jun-2019. [Online]. Available: https://towardsdatascience.com/predicting-nba-rookie-stats-with-machine-learning-28621e49b8a4. [Accessed: 17-Dec-2019].
