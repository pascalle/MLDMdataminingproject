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



####Couldn't finish###########################################
###########################################
###########################################
##Parameter tuning
#printcp(fit) #look for lowest xerror to find best cp value,
#or use lowest xerror plus its stdeviation = num, then look for max xerror that is lower than num
#plotcp(fit) #gives graph to find optimal number
##pruning the tree based off of the cp we find
#prune.fit = prune(fit, cp = 0.014815) #use cp value found above
#rpart.plot(prune.fit, cex = .5, extra = 4)

#can also use the following code to automatically get the cp values
#fit$cptable[which.min(fit$cptable[,”xerror”]),”CP”]

#Prune the tree to create an optimal decision tree :
# ptree<- prune(tree,
# cp= tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"])
# fancyRpartPlot(ptree, uniform=TRUE,
# main="Pruned Classification Tree")
#
#parameter tuning
##cp - complexity parameter, low cp value gives very accurate model gives high variance high complexity, high cp value gives low number of splits, low bias, low complexity
##minsplit - the minimum number of observations in a node for it to split. If <5 nodes won't split when it has <5 obs
##minbucket - minimum number of obs for a terminal (leaf) node
##maxdepth - maximum depth of the tree
##xval - does x fold crossvalidation

#if I want to get the observations falling into a particular node
#node2 <- subset.rpart(tree.fit, df, 2)



#########################################################