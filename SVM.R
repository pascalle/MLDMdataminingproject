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


