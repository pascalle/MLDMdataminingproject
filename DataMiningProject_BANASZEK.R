#MLDM1 Data Mining Project
#Pascalle Banaszek
#Apr 14, 2021


#############Importing and Loading Data###########################
#dataset: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
library(dplyr)
library(caTools)
library(rpart)
library(rpart.plot)
library(xtable)
library(caret)
library(e1071)
library(ROSE)

setwd("/Users/bananasacks/Desktop/MLDM Courses/Semester 2/Data Mining/Muhlenbach/Project/")
stroke <- read.csv("healthcare-dataset-stroke-data.csv")
head(stroke)
str(stroke) #checks class of all the columns
summary(stroke)
colSums(is.na(stroke))
xtable(summary(select(stroke,c(stroke,age, avg_glucose_level,bmi,hypertension,heart_disease))), type="latex")

#############Data Preprocessing##########################
#remove N/A
stroke <- subset(stroke, bmi!="N/A")
#convert 0/1 to factors and bmi as numeric
stroke <- mutate(stroke, 
                    #stroke = as.factor(bmi),
                    #hypertension = as.factor(hypertension),
                    #heart_disease = as.factor(heart_disease),
                    bmi = as.numeric(bmi))
#shuffling the dataset
shuffle_index <- sample(1:nrow(stroke))
stroke <- stroke[shuffle_index, ]
#########################################################
 
#############Function: Spitting into Training and Test Sets########
#split data into testing and train
traintestsplit <- function(datset, ratio) {
  set.seed(4269)
  sample = sample.split(datset, SplitRatio = ratio)
  train = subset(datset, sample == TRUE)
  test  = subset(datset, sample == FALSE)
  list = list("train" = train, "test" = test)
  return(list)
}
#########################################################


#############Variable Analysis###########################
#using unique(stroke$column_name)
#data.frame(table(stroke$column_name)) - to check balances/imbalances of sets
#sum(stroke$bmi == "N/A") - to find character N/A variables

########Continuous Variables##
#age
#avg_glucose_level
#bmi

########Binary Variables##
#hypertension [mean .097]
#heart_disease [mean .054]
#stroke [mean .048]

########Categorical Variables##
#gender [Male, Female, Other]
#ever_married [Yes, No]
#work_type [Private, Self-employed, Govt_job, children, Never_worked]
#residence_type [Urban, Rural]
#smoking_status [formerly smoked, never smoked, smokes, Unknown]

######Other options######
#examine correlations between different features

#check correlations between numeric variables
str(stroke)
str(num_stroke)
num_stroke <- select(stroke, c(age, avg_glucose_level, bmi))
cor_stroke <- cor(num_stroke)
cor_stroke
#non-normalized correlation table
xtable(cor_stroke, type="latex")

#graph pairs scatter grouping stroke by color
num_stroke <- select(stroke, c(age, avg_glucose_level, bmi, stroke))
group <- NA
group[num_stroke$stroke == 1] <- 2
group[num_stroke$stroke == 0] <- 1
pairs_stroke <- pairs(select(num_stroke, -c(stroke)), pch = 19, col = c("cornflowerblue", "purple")[group])

#normalizing variables between 0 and 1
norm_num_stroke <- preProcess(num_stroke, method=c("range"))
norm_num_stroke <- predict(norm_num_stroke, num_stroke)
norm_cor_stroke <- cor(norm_num_stroke)
norm_cor_stroke
#normalized correlation table
xtable(norm_cor_stroke, type="latex")


#normalizing changes nothing as expected, but gives us an opportunity to compare
#properly to stroke variable
group <- NA
group[num_stroke$stroke == 1] <- 2
group[num_stroke$stroke == 0] <- 1
pairs_stroke <- pairs(select(norm_num_stroke, -c(stroke)), pch = 19, col = c("cornflowerblue", "purple")[group])


#########################################################

#############Data Preprocessing##########################
##convert categorical into expanded columns
##standardize data
##create datasets of just continuous data for SVM

#########################################################



########### SVM ##############################################
#only use continuous variables: age, avg_glucose_level, bmi
#svmStroke <- subset(stroke, select = c(stroke, age, avg_glucose_level, bmi))
#svmStroke <- mutate(svmStroke, 
#       stroke = factor(stroke, levels = c(0, 1), labels = c('No', 'Yes')),
#       age = as.numeric(age),
#       avg_glucose_level = as.numeric(avg_glucose_level),
#       bmi = as.numeric(bmi))


#########################################################





####Random Forest########################################
#https://towardsdatascience.com/random-forest-in-r-f66adf80ec9
#https://www.kaggle.com/csyhuang/predicting-chronic-kidney-disease
#random forest
rf <- randomForest(
  num ~ .,
  data=train)
#predictions
pred = predict(rf, newdata=test[-14])
#confusion matrix
cm = table(test[,14], pred)
#########################################################


#######################SVM###############################
#https://www.datacamp.com/community/tutorials/support-vector-machines-r
#Can do a 2D SVM with age and BMI and color code based on stroke or not
#Maybe can separate with men and women
#Then can do SVM with all features
#
#########################################################



#########################################################
##Split train and test
##Fit model on train
##k fold cross validation
##https://www.geeksforgeeks.org/cross-validation-in-r-programming/
##Gridsearch cross validation to tune parameters
##
##Check effectiveness of model on test set
##confusion matrix, precision, recall
##bias versus variance?


#########################################################
#Maybe I can make a prediction tool that you put in your age, weight, smoking status etc
#it calculates your BMI and then predicts your likelihood to have a stroke

