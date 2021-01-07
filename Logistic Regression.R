library(titanic)
library(splines)
library(broom)
#First we process the titatnic_train dataset to make it a little more logistic regression friendly
titanic <- titanic_train %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare) %>%
  mutate(Survived = factor(Survived),
         Pclass = factor(Pclass),
         Sex = factor(Sex))



#making the sex column binary with male = 1
for(i in 1:nrow(titanic))
  {
       if(titanic$Sex[i] == "male")
         {
               titanic$sex_binary[i] <- 1
            }
      if(titanic$Sex[i] == "female")
         {
               titanic$sex_binary[i] <- 0
          }
   }
#First we need to perform some basic data preparation by removing NA's and re-coding factors to make it easier for us to keep track of.
titanic <- na.omit(titanic)


#Breaking down data into train and test split
titanic <- titanic[,c(1,2,4,5,6,7,8)]
set.seed(2)
vec <- sample(c(1:nrow(titanic)), (nrow(titanic)/3),replace = FALSE)
test <- titanic[vec,]
train <- titanic[-vec,]
#Now we can actually generate the model
model <- glm(Survived ~., data = train, 
             family = binomial)

#Eliminating collinearity with backward selection; we know something is wrong because Fare is not in the model, even though exploratory data analysis
#clearly shows a relationship between fare and survival
model <- step(model)

#We need to generate a vector of probabilities associated with each row in order to perform daignostics ensuring we are meeting the assumptions of
#logistic regression
probabilities <- predict(model, type = "response")


#First we check the assumption of linearity between continuous predictors and the outcome logit.  To do this we must select our numeric type predictors.
my_data <- train %>% select(Age, Fare)
numeric_predictors <- colnames(my_data)

#Now we seperate out the predictors we have chosen along with their predicted percentage and logit
mydata <- my_data %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

#Now we create a visualization of the predictors versus associate logit values to asses our assumption of linearity
ggplot(mydata, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")

#Before we deal with the non-linearity of the age model, we have to investigate the possibility of influential outlier values.
#First, let's look at Cook's distance for our most visaully noticeable possible outliers
plot(model, which = 4,id.n = 8)

#Now we pull the data for these points
model.data <- augment(model) %>% 
  mutate(index = 1:n())
model.data %>% top_n(8, .cooksd)

ggplot(model.data, aes(index, .std.resid)) + 
  geom_point(aes(color = Survived), alpha = .5) +
  theme_bw()+
  xlab("Index")+
  ylab("Standard Residual")+
  ggtitle("Standardized Residuals of Model")+
  ylim(c(-3,3))+
  scale_color_discrete(name = "Survival Status", labels = c("Did Not Survive", "Survived"))

#We notice that we have no standardized residuals over absolute value of 3, so we can be confident that we do not have influential outliers
##########Introduction of Splines############
#With the fare predictor, we could argue that it approximates a linear relationship with the logit closely enough, but it is harder to make the case for
#the age predictor.  We'll try to use natural cubic splines to account for this non-linearity.

#Optimizing the number of degrees of freedom via resampling; we suspect there would be up to
# 3 regions (df = 2) for age based on graphics from exploratory phase
#We'll add one in the interest of exploring additional possibilities

#Histrogram to provide some additional intuition about how many degrees of freedom may be appropriate
titanic %>% ggplot(aes(x = Age))+
  geom_density()+
  xlab("Age")+
  ylab("Density")+
  ggtitle("Estimating Initial Degrees of Freedom: Age")


#Randomly Resampling to find out what the ideal number of degrees of freedom for the Age spline is.  We randomly select 1/3 of the data to test and 2/3 to train
#For each iteration, each number of degrees of freedom (1-6) will be used to create a model and accuracy will be calculated.  We define accuracy as simply a 
#result that matches a prediction.  Inaccuracies are results that do not match a prediction; we do not distinguish between false positives and false negatives
#for these purposes.  The degree of freedom that is most accurate against the test set is stored and the loop starts over.  We repeat this 1,000 times, then analyze
#which degree of freedom was most frequently the most accurate.
set.seed(1)
dist <- data.frame()
for(k in 1:1000)
{
  vec <- sample(c(1:nrow(titanic)), (nrow(titanic)/3),replace = FALSE)
  test <- titanic[vec,]
  train <- titanic[-vec,]
  
  test <- test %>% mutate(survived_numeric = as.numeric(Survived)-1)
  
  accuracy_data <- data.frame()
  
  for(i in 1:4)
  {
      
      model <- glm(Survived ~ SibSp + ns(Age, df = i) + Pclass + Parch + Fare,
                   data = titanic,
                   family = binomial)
      model <- step(model, trace = FALSE)
      
      probabilities <- predict(newdata = test, model, type = "response")
      df <- test %>% mutate(predicted_percent_survival = predict(newdata = test, model, type = "response"))
      df$predicted_percent_survival <- round(df$predicted_percent_survival)
      df <- df %>% mutate(diff = abs(survived_numeric-predicted_percent_survival))
      accuracy_data[i,1] <- 1-(sum(df$diff)/nrow(df))
      accuracy_data[i,2] <- i
      df <- NA
    
  }
  
  colnames(accuracy_data)[1] = "Accuracy"
  colnames(accuracy_data)[2] = "i"
  
  accuracy_data <- na.omit(accuracy_data)
  
  minimum<-accuracy_data %>% filter(Accuracy==max(Accuracy))
  dist[k,1] <- (minimum[[1,2]])
  dist[k,2] <- minimum[[1,1]]
}

colnames(dist)[1] <- "i"
colnames(dist)[2] <- "accuracy"
View(dist)

#After resampling and testing different combinations of degrees of freedom 1,000 times, the use of 3 degrees of freedom clearly produces the most accurate 
#test results the most frequently.  In the table "data", we see each degree of freedom and we see how many out of
#the 1,000 models had the most accurate results against the randomly selected test set with degree of freedom.
data <- dist %>% mutate(count = 1) %>% group_by(i) %>% summarise(accuracy = mean(accuracy), count = sum(count))

#Now that we know how many degrees of freedom we want to use, we can continue building our model.  We now address collinearity possibilities with
#a backward selection process similar to what we used in linear modeling.
model_02 <- glm(Survived ~ SibSp + ns(Age, df = 3) + Pclass + Parch + Fare,
             data = titanic,
             family = binomial)
model_02 <- step(model_02, trace = FALSE)

summary(model_02)
#In the model summary, we see that having more siblings aboard decreases the survival probability.  We notice that age also lowers probability of 
#surviving.  This matches intuition and the graphics used in our previous analysis ("Additional Data Visaulizations", same repo). We also note that 
#Passenger class decreases probability of surviving.  This makes sense because a higher class means lower levels of accomodation that we saw in the 
#"Additional Data Visualizations" section is related to survival in this way.  Finally, we notice that fare is positively related to survival, which
#also matches our intuition and the "Additional Data Visualizations" graphic.

#Here we can see where the knots related to age are actually positioned.  Notice that the first boundary to the first knot has the coefficient closest to 
#zero among the age coefficients.  Also notice that there is a significant drop to the function between the middle two knots, and that the function between
#the third knot and the boundary is in the middle.  The transitions between these values align with our knowledge that the youngest age group was most liekly to 
#survive, the oldest age group second most likely, and the middle age group least likely.
attr(model_02$model$`ns(Age, df = 3)`, "knots")

#########Cross-Validation###################
#now that we have a model, let's use 3-fold cross validation to test it out

#first we shuffle our rows randomly
set.seed(3)
rows <- sample(nrow(titanic))
shuffled <- titanic[rows,]

#Then we split our data set for 3-fold cross validation
t <- nrow(shuffled)/3
df1 <- shuffled[1:t,]
df2 <- shuffled[(t+1):(2*t),]
df3 <- shuffled[((2*t)+1):(3*t),]

#Beginning first fold
train1 <- bind_rows(df1,df2)
test1 <- df3

model1 <- glm(Survived ~SibSp + ns(Age, df = 3) + Pclass + Fare,
              data = train1, 
             family = binomial)
test1 <- test1 %>% mutate(predicted_percent_survival = predict(model1, newdata = test1, type = "response"))

test1$predicted_percent_survival <- round(test1$predicted_percent_survival)
test1 <- test1 %>% mutate(survived_numeric = as.numeric(Survived)-1)
test1 <- test1 %>% mutate(diff = abs(survived_numeric-predicted_percent_survival))
accuracy1 <- 1-(sum(test1$diff)/nrow(test1))
binom.test((nrow(test1)*accuracy1), nrow(test1), alternative = "two.sided")

#Beginning second fold
train2 <- bind_rows(df1,df3)
test2 <- df2

model2 <- glm(Survived ~SibSp + ns(Age, df = 3) + Pclass + Fare,
              data = train2, 
              family = binomial)
test2 <- test2 %>% mutate(predicted_percent_survival = predict(model2, newdata = test2, type = "response"))

test2$predicted_percent_survival <- round(test2$predicted_percent_survival)
test2 <- test2 %>% mutate(survived_numeric = as.numeric(Survived)-1)
test2 <- test2 %>% mutate(diff = abs(survived_numeric-predicted_percent_survival))
accuracy2 <- 1-(sum(test2$diff)/nrow(test2))

#beginning third fold
train3 <- bind_rows(df2,df3)
test3 <- df1

model3 <- glm(Survived ~SibSp + ns(Age, df = 3) + Pclass + Fare,
              data = train3, 
              family = binomial)
test3 <- test3 %>% mutate(predicted_percent_survival = predict(model3, newdata = test3, type = "response"))

test3$predicted_percent_survival <- round(test3$predicted_percent_survival)
test3 <- test3 %>% mutate(survived_numeric = as.numeric(Survived)-1)
test3 <- test3 %>% mutate(diff = abs(survived_numeric-predicted_percent_survival))
accuracy3 <- 1-(sum(test3$diff)/nrow(test3))

binom.test((nrow(test3)*accuracy3), nrow(test3), alternative = "two.sided")

#Note that all accuracy values have a 2-sided p-value well below 0.01 indicating the model allows us to reject the null hypothesis that we do not gain any predictive
#power with the model

#Evaluation
#in general, we can feel confident that our model predicts who would survive or not survive the Titanic wreck with ~70.4% accuracy
mean_accuracy <- (accuracy1 + accuracy2 + accuracy3)/3
mean_accuracy
###Don't forget to eliminate significant outliers as an assumption; explicitly list assumptions at beginning