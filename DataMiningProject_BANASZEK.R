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


#####DECISION TREE###############################
####Decision Trees########################################

treeStroke <- select(stroke, -c(id))

treeStroke <- mutate(treeStroke, 
                     stroke = factor(stroke, levels = c(0, 1), labels = c('No', 'Stroke')),
                     age = as.numeric(age),
                     hypertension = as.factor(hypertension),
                     heart_disease = as.factor(heart_disease),
                     #ever_married = factor(ever_married, levels = c('No', 'Yes'), labels = c(0, 1)),
                     #work_type = factor(work_type, levels = c('Private', 'children', 'Self-employed', 'Govt_job', 'Never_worked') , labels = c(1, 2, 3, 4, 5)),
                     #Residence_type = factor(Residence_type, levels = c('Rural', 'Urban'), labels = c(0, 1)),
                     bmi = as.numeric(bmi),
                     smoking_status = factor(smoking_status, levels = c("formerly smoked", "Unknown", "never smoked", "smokes"), labels = c(1, 2, 3, 4)))



#Do not need to standardize, can mix continuous and categorical variables
tree <- traintestsplit(treeStroke,2/3)
#tree_train <- select(tree$train, -c(stroke))
#tree_test <- select(tree$train, -c(stroke))


tree_train <- tree$train
tree_test <- tree$test

fit <- rpart(stroke~., data = tree_train, cp=.004, method = 'class')#, minsplit=5, min_bucket=5)
rpart.plot(fit, cex = .5, extra = 106, type=2)
#rpart.plot(fit, box.palette="RdBu", shadow.col="gray", nn=TRUE)

##Evaluating the model
tree_predict <-predict(fit, tree_test, type="class")
table(tree_test$stroke, tree_predict)


accuracy.meas(tree_test$stroke, tree_predict)
roc.curve(tree_predict, tree_test$stroke, plotit = F)
confusionMatrix(table(tree_predict, tree_test$stroke))
xtable(table(tree_predict, tree_test$stroke), type="latex")

###############################################
#########################################################



########### SVM ##############################################
##Run Beginning of Data Mining Project
#Lines 8 through 48
########### SVM ##############################################
#only use continuous variables: age, avg_glucose_level, bmi
svmStroke <- subset(stroke, select = c(stroke, age, avg_glucose_level, bmi))
#make target variable as a factor
svmStroke <- mutate(svmStroke, 
                    stroke = factor(stroke))
str(svmStroke)

####SPLTTING TRAINING AND TEST SETS#####################################
svm <- traintestsplit(svmStroke,.75)
svmtrain <- subset(svm$train)
svmtest <- subset(svm$test)

####STANDARDIZATION#####################################################
#normalize the variables here, everything between 0 and 1 
#train_norm <- preProcess(svmtrain, method=c("range"))
#train_norm <- predict(train_norm, train)
#I shouldn't have to normalize the test
#test_norm <- preProcess(test, method=c("range"))
#test_norm <- predict(test_norm, test)

#decided to standardize instead
train_std <- preProcess(svmtrain, method=c("center", "scale"))
train_std <- predict(train_std, svmtrain)

test_std <- preProcess(svmtest, method=c("center", "scale"))
test_std <- predict(test_std, svmtest)

#####BALANCING THE DATA####################################################
table(train_std$stroke)
#over sampling
data_balanced_over <- ovun.sample(stroke~., data=train_std, method = "over",N = 7052)$data
table(data_balanced_over$stroke)

#under sampling
data_balanced_under <- ovun.sample(stroke~., data=train_std, method = "under",N = 312, seed=42)$data
table(data_balanced_under$stroke)

#########################################################


#####RUNNING THE SVM and TIMING####################################################
#UNDER SAMPLING radial svm, since I standardized before the balancing, I don't need to do it within the svm
set.seed(4269)
start_time <- Sys.time()
rbfunder.tune <- tune.svm(stroke~., data=data_balanced_under, scale = FALSE, kernel="radial", 
                          gamma = c(.0001, .001,.01,.1,1.10,100,200,500,1000), cost = c(.0001,.005,.01,.05,1,2,5,10,100))
end_time <- Sys.time()
under_time <- end_time - start_time

summary(rbfunder.tune)
bestunder.model <- rbfunder.tune$best.model
tuneunder.test <- predict(bestunder.model, newdata=test_std)#newdata=svmtest)
table(tuneunder.test, test_std$stroke)#svmtest$stroke)
table(tuneunder.test)

####ACCURACY CHECK
#Checking the accuracy of the predictions
#To check accuracy, ROSE package has a function names accuracy.meas, 
#it computes important metrics such as precision, recall & F measure.
accuracy.meas(test_std$stroke, tuneunder.test)
roc.curve(tuneunder.test, test_std$stroke, plotit = F)
confusionMatrix(table(tuneunder.test, test_std$stroke))
xtable(table(tuneunder.test, test_std$stroke), type="latex")


#OVER SAMPLING radial svm, since I standardized before the balancing, I don't need to do it within the svm
set.seed(4269)
start_time <- Sys.time()
rbfover.tune <- tune.svm(stroke~., data=data_balanced_over, scale = FALSE, kernel="radial", 
                         gamma = c(.001,.01,.1,1.10,100), cost = c(.0001,.005,.01,.05,1,5,10))
end_time <- Sys.time()
over_time <- end_time - start_time
#over_time
#Time difference of 15.79123 mins

summary(rbfover.tune)
bestover.model <- rbfover.tune$best.model
tuneover.test <- predict(bestover.model, newdata=test_std) #remove stroke from here?
table(tuneover.test, test_std$stroke)

####ACCURACY CHECK
#Checking the accuracy of the predictions
#To check accuracy, ROSE package has a function names accuracy.meas, 
#it computes important metrics such as precision, recall & F measure.
accuracy.meas(test_std$stroke, tuneover.test)
roc.curve(tuneover.test, test_std$stroke, plotit = F)
confusionMatrix(table(tuneover.test, test_std$stroke))
table(tuneover.test)
table(test_std$stroke)
xtable(table(tuneover.test, test_std$stroke), type="latex")


#########################################################
