# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from IPython.display import display
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/TeamPerformance0919.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
PastPerformanceData = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# Create a view or table
temp_table_name = "TeamPerformance0919_csv"
PastPerformanceData.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `TeamPerformance0919_csv`

# COMMAND ----------

read_team_name = sqlContext.read.format('com.databricks.spark.csv') \
    .options(header='true', inferschema='true',sep=',') \
    .load('./FileStore/tables/season_1920-e7f15.csv')

temp_table_name = "1920seasondata"
read_team_name.createOrReplaceTempView(temp_table_name)
read_team_names = read_team_name.toPandas()

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from `1920seasondata`

# COMMAND ----------

team_names_list = spark.sql("SELECT distinct(HomeTeam) from 1920seasondata")
display(team_names_list)

# COMMAND ----------

team_name_list = read_team_names['HomeTeam']
team_name=[]
for teams in team_name_list:
    if teams not in team_name:
        team_name.append(teams)
print("\n\nTeams in Season: 2019-2020")

teams_data_frame = pd.DataFrame(team_name, columns=["Team Names"])
display(teams_data_frame)
print(teams_data_frame[teams_data_frame['Team Names'] == "Man United"])
print(teams_data_frame[teams_data_frame['Team Names'] == "Brighton"])

# COMMAND ----------

PastPerformanceDataPanda = PastPerformanceData.toPandas()
PastPerformanceDataPanda.info()

# COMMAND ----------

n_matches = PastPerformanceDataPanda.shape[0] #[0] for X-axis
n_features = PastPerformanceDataPanda.shape[1] - 1  #[1] for Y-axis (total features - Labels to be determined)

print(n_features)

n_homewins = len(PastPerformanceDataPanda[PastPerformanceDataPanda.FTR == 'H'])
win_rate = (float(n_homewins)/(n_matches))*100

print("Total no of matches: {}".format(n_matches))
print("Number of Features: {}".format(n_features))
print("Number of matches won by HOME: {}".format(n_homewins))
print("Win rate of HOME team: {}".format(win_rate))

# COMMAND ----------

def getTeamData(teamName):
    print("\n======================= "+ teamName + " =====================\n")
    
    #Num of goals in wins and looses
    gamesHome = PastPerformanceDataPanda[PastPerformanceDataPanda['HomeTeam']== teamName]
    totalGoalsScored = gamesHome['FTHG'].sum()
    
    gamesAway = PastPerformanceDataPanda[PastPerformanceDataPanda['AwayTeam'] == teamName]
    totalGames = gamesHome.append(gamesAway)
    numGames = len(totalGames.index)
    totalGoalsScored += gamesAway['FTAG'].sum() 
    
    
    #total goals allowed 
    totalGoalsAllowed = gamesHome['FTHG'].sum()
    totalGoalsAllowed += gamesAway['FTAG'].sum()
    
    #discipline TOTAL RED AND YELLOW CARDS
    totalYellowCards = gamesHome['HY'].sum()
    totalYellowCards += gamesAway['AY'].sum()
    
    totalRedCards = gamesHome['HR'].sum()
    totalRedCards += gamesAway['AR'].sum()
    
    
    #total Fouls
    totalFouls = gamesHome['HF'].sum()
    totalFouls += gamesAway['AF'].sum()
    
    
    #total Corners
    totalCorners = gamesHome['HC'].sum()
    totalCorners += gamesAway['AC'].sum()
    
    
    #shots per game (SPG) = totalshots / totalgames
    totalShots = gamesHome['HS'].sum()
    totalShots += gamesAway['AS'].sum()
    
    #avg shots allowed per game
    totalShotsAgainst = gamesHome['AS'].sum()
    totalShotsAgainst += gamesAway['HS'].sum()
    if numGames != 0:
        HSPG = totalShots / numGames #HomeShotsPerGame
        ASPG = totalShotsAgainst / numGames #AwayShotsPerGame
        display("HSPG: {}".format(HSPG))
        display("ASPG: {}".format(ASPG))
    
    #games won percentage= GamesWon / numGames
    gamesWon = totalGames[totalGames['FTR']== "H"]
    gamesLost = totalGames[totalGames['FTR'] == "A"]
    gamesDraw = totalGames[totalGames['FTR'] == "D"]
    numGamesWon = len(gamesWon.index)
    numGamesLost = len(gamesLost.index)
    numGamesDraw = len(gamesDraw.index)
    
    if numGames != 0:
        gamesWonPercent = numGamesWon / numGames
        gamesLostPercent = numGamesLost / numGames
        gamesDrawPercent = numGamesDraw / numGames 
    
    print("Games Win Percent: {}".format(gamesWonPercent))
    print("Games Loose Percent: {}".format(gamesLostPercent))
    print("Games Draw Percent: {}".format(gamesDrawPercent))
    
    
    #Total shots on target:
    totalShotsOnTarget = gamesHome['HST'].sum()
    totalShotsOnTarget += gamesAway['AST'].sum()
    
    #GoalSaves
    goalSaves = totalShotsOnTarget - totalGoalsAllowed
    
    #Goal Save Percentage
    if totalShotsOnTarget != 0:
        goalSavesPercent = goalSaves / totalShotsOnTarget
        
    #Goal Save Ratio
    if goalSaves != 0:
        saveRatio = totalShotsOnTarget / goalSaves
    
    #Goal scoring Percent
    if totalShots != 0 :
        scoringPercent = (totalShots - totalGoalsScored)/totalShots
    
    #Goal scoring Ration
    if totalGoalsScored != 0:
        scoringRatio = totalShotsOnTarget / totalGoalsScored
        
    if numGames == 0: 
        gamesWon = 0
        gamesLost = 0
        gamesDraw = 0 
        totalGoalsScored = 0 
        totalShotsOnTarget = 0 
        totalGoalsAllowed = 0 
        totalYellowCards = 0 
        totalRedCards = 0 
        totalFouls = 0 
        totalCorners = 0 
        totalShots = 0 
        totalShotsAgainst = 0 
        HSPG = 0 #HomeShotsPerGame 
        ASPG = 0 #AwayShotsPerGame 
        goalSaves = 0 
        goalSavesPercent = 0 
        scoringPercent = 0 
        saveRatio = 0 
        scoringRatio = 0
    
    return [teamName, totalGoalsScored, totalShotsOnTarget, totalGoalsAllowed, 
            totalYellowCards, totalRedCards,totalFouls, totalCorners, 
            totalShots, totalShotsAgainst, HSPG, ASPG, goalSaves, goalSavesPercent, scoringPercent,
            saveRatio, scoringRatio]


# COMMAND ----------

getTeamData('Man United')

# COMMAND ----------

newStatList = []
for team in team_name:
    team_vector = getTeamData(team) 
    newStatList.append(team_vector)
    
teamStats = pd.DataFrame(newStatList, columns=['TeamName', 'totalGoalsScored', 'totalShotsOnTarget', 'totalGoalsAllowed', 
            'totalYellowCards', 'totalRedCards','totalFouls', 'totalCorners', 
            'totalShots', 'totalShotsAgainst', 'HSPG', 'ASPG', 'goalSaves', 'goalSavesPercent', 'scoringPercent',
            'saveRatio', 'scoringRatio'])

# COMMAND ----------

teamstats = sqlContext.createDataFrame(teamStats)
teamstats.createOrReplaceTempView("teamstats")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from teamstats

# COMMAND ----------

# MAGIC %sql
# MAGIC select TeamName, totalGoalsScored,  totalShots from teamstats

# COMMAND ----------

# MAGIC %sql
# MAGIC select TeamName, totalFouls,  totalRedcards, totalYellowCards from teamstats

# COMMAND ----------

totalShotsPlot = sns.barplot(teamStats.TeamName, teamStats.totalShots) 
for item in totalShotsPlot.get_xticklabels():
    item.set_rotation(90)

# COMMAND ----------

scoringRatioPlot = sns.barplot(teamStats.TeamName, teamStats.scoringRatio) 
for item in scoringRatioPlot.get_xticklabels():
    item.set_rotation(90)

# COMMAND ----------

PastPerformanceDataPanda.isnull().sum()

# COMMAND ----------

teamStats.isnull().sum()

# COMMAND ----------

from pandas.plotting import scatter_matrix
scatter_matrix(PastPerformanceDataPanda[['FTHG','FTAG']],figsize=(20,20))

# COMMAND ----------

for team in team_name:
  filteredData = PastPerformanceDataPanda[(PastPerformanceDataPanda.HomeTeam.isin(team_name))]
  PastPerformanceDataPanda = filteredData[(filteredData.AwayTeam.isin(team_name))]

# COMMAND ----------

# Separate into feature set and target variable
# Data loaded into x axis for training
X_all = PastPerformanceDataPanda.drop(['FTR'],1)   
# Data loaded into y Axis for training
y_all = PastPerformanceDataPanda['FTR']
# Data loaded into y axis for training
Z_all = X_all.drop(['Date','HTR','Referee'],1)

#print(X_all)
#print(y_all)
# print(Z_all)

# Standardising the data.
from sklearn.preprocessing import scale
cols = [['FTHG','FTAG','HTHG','HTAG','HS','AS','HST','HF','AF','HY','AY','HR','AR','HC','AC','AST','HC','AC']]
for col in cols:
    X_all[col] = scale(X_all[col])

# COMMAND ----------

Z_all.tail()

# COMMAND ----------

#we want continous vars that are integers for our input data, so lets remove any categorical vars
def preprocess_features(Z):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = Z.index)

    # Investigate each feature column for the data
    for col, col_data in Z.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revisedd columns
        output = output.join(col_data)
    
    return output

# COMMAND ----------

Z_all = preprocess_features(Z_all)
print ("Processed feature columns ({} total features):\n{}".format(len(Z_all.columns), list(Z_all.columns)))

# COMMAND ----------

# Show the feature information by printing the first five rows
Z_all = sqlContext.createDataFrame(Z_all)
Z_all.createOrReplaceTempView("Z_all")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from Z_all;

# COMMAND ----------

from sklearn.model_selection import train_test_split

Z_all = Z_all.toPandas();
# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(Z_all, y_all, 
                                                    test_size = 50,
                                                    random_state = 2,
                                                    stratify = y_all)

# COMMAND ----------

#for measuring training time
from time import time 
# F1 score (also F-score or F-measure) is a measure of a test's accuracy. 
#It considers both the precision p and the recall r of the test to compute 
#the score: p is the number of correct positive results divided by the number of 
#all positive results, and r is the number of correct positive results divided by 
#the number of positive results that should have been returned. The F1 score can be 
#interpreted as a weighted average of the precision and recall, where an F1 score 
#reaches its best value at 1 and worst at 0.
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))
    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, average='macro'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print (f1, acc)
    print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
    f1, acc = predict_labels(clf, X_test, y_test)
    print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))

# COMMAND ----------

# Initialize the three models (XGBoost is initialized later)
clf_A = LogisticRegression(random_state = 50)
clf_B = SVC(random_state = 912, kernel='rbf')
clf_C = MultinomialNB()
#Boosting refers to this general problem of producing a very accurate prediction rule 
#by combining rough and moderately inaccurate rules-of-thumb

train_predict(clf_A, X_train, y_train, X_test, y_test)
print ('')
train_predict(clf_B, X_train, y_train, X_test, y_test)
print ('')
train_predict(clf_C, X_train, y_train, X_test, y_test)
print ('')

# COMMAND ----------

model = LogisticRegression()
model.fit(X_train, y_train)

# COMMAND ----------

model.predict(X_test)

# COMMAND ----------

predictedProbability = model.predict_proba(X_test)
predictedProbability = pd.DataFrame(predictedProbability, columns=['Away Team','Draw','Home Team'])

display((predictedProbability *100).head(10))

# COMMAND ----------

fixtures = spark.read.csv('/FileStore/tables/fixtures.csv', header="true", inferSchema="true").toPandas()
# fixtures = fixtures[pd.isnull(fixtures['Result'])] #drop all the rows having result certain values
fixtures = fixtures.drop(['Round Number','Date','Location','Result'],1)
fixtures.columns = ['HomeTeam','AwayTeam']
fixtures['FTHG']= 0
fixtures['FTAG'] =0 
fixtures['HTHG'] = 0
fixtures['HTAG'] = 0
fixtures['HS'] = 0
fixtures['AS'] = 0
fixtures['HST'] = 0
fixtures['AST'] = 0
fixtures['HF'] = 0
fixtures['AF'] = 0
fixtures['HC'] = 0
fixtures['AC'] = 0
fixtures['HY'] = 0
fixtures['AY'] = 0
fixtures['HR'] = 0
fixtures['AR'] = 0

# COMMAND ----------

fixture = sqlContext.createDataFrame(fixtures)
fixture.createOrReplaceTempView("fixtures")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from fixtures;

# COMMAND ----------

preprocessedFixtures = preprocess_features(fixture.toPandas())
print ("Processed feature columns ({} total features):\n{}".format(len(preprocessedFixtures.columns),
                                                                   list(preprocessedFixtures.columns)))

# COMMAND ----------

model.predict(preprocessedFixtures)

# COMMAND ----------

fixtures['Result Predicted'] = model.predict(preprocessedFixtures)

Result = fixtures.drop(['FTHG','FTAG','HTHG','HTAG','HS','AS','HST','HF','AF','HY',
                        'AY','HR','AR','HC','AC','AST','HC','AC'],1)

Result = sqlContext.createDataFrame(Result)
Result.createOrReplaceTempView("Result")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from Result;

# COMMAND ----------

fixturePredictedProbability = model.predict_proba(preprocessedFixtures) *100
fixturePredictedProbability = pd.DataFrame(fixturePredictedProbability,
                                columns=['Away win %','Draw %','Home win %'])

display(fixturePredictedProbability)

# COMMAND ----------

final = pd.concat([Result.toPandas(), fixturePredictedProbability], axis = 1)
matchPrediction = sqlContext.createDataFrame(final)
matchPrediction.createOrReplaceTempView("matchPrediction")

display(final)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from matchPrediction

# COMMAND ----------

readFixtures = spark.read.csv('/FileStore/tables/fixtures.csv', header="true", inferSchema="true").toPandas()
exportToFixtures = final.drop(['HomeTeam','AwayTeam'],1)

PredictedResultWithFixtureData = pd.concat([readFixtures,exportToFixtures], axis = 1)
finalResults = sqlContext.createDataFrame(PredictedResultWithFixtureData)
#PredictedResultWithFixtureData.to_csv('./dataset/Final-Results/Predicted_Result_With_Fixture_Data.csv')

finalResults.createOrReplaceTempView("finalResults")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from finalResults

# COMMAND ----------

for teams in teams_data_frame:
    teams_data_frame["points"]=0

results = PredictedResultWithFixtureData["Result Predicted"]
home = PredictedResultWithFixtureData["Home Team"]
away = PredictedResultWithFixtureData["Away Team"]
points = teams_data_frame["points"]

h = "H"
a = "A"
d = "D"
i = 0

for result in results:
    j = 0
    if(result==h):
        for team in teams_data_frame["Team Names"]:
            if(home[i]==team):
                teams_data_frame["points"][j] = teams_data_frame["points"][j] + 3
            j = j+1;
    elif(result==a):
        for team in teams_data_frame["Team Names"]:
            if(away[i]==team):
                teams_data_frame["points"][j] = teams_data_frame["points"][j] + 3
            j = j+1;
    elif(result==d):
        for team in teams_data_frame["Team Names"]:
            if(away[i]==team):
                teams_data_frame["points"][j] = teams_data_frame["points"][j] + 1
            if(home[i]==team):
                teams_data_frame["points"][j] = teams_data_frame["points"][j] + 1
           
            j = j+1;
    i=i+1;
teams_data_frame = sqlContext.createDataFrame(teams_data_frame)
teams_data_frame.createOrReplaceTempView("teams_data_frame")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from teams_data_frame
