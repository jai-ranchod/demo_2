#####Intitial Processing and Definitions#####
#Here we use a logistic regression model as a means of binary classification to predict who will survive and who will not survive the titanic wreck
library(titanic)
library(splines)
library(broom)
library(dplyr)
library(ggplot2)
library(tidyr)
#First we process the titatnic_train dataset to make it a little more logistic regression friendly
titanic <- titanic_train %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare) %>%
  mutate(Survived = factor(Survived),
         Pclass = factor(Pclass),
         Sex = factor(Sex))

#making the sex column binary with male = 1
titanic$sex_binary <- as.integer(titanic$Sex == "male")


#Survived - 0=Did not survive, 1 = survived
#Pclass - Passenger class; 1st class being the most expensive, 2nd class, is cheaper, 3rd class is cheapest
#Sex - Sex of passenger, female or male
#Age - Age of passenger; notice occasional missing data here
#SibSp - Number of siblings/spouses on board
#Parch - Number of Parents/Children on board
#Fare - Passage fare in 1912 British Pounds
#sex_binary - a different expression of Sex

#First we need to perform some basic data preparation by removing NA's and re-coding factors to make it easier for us to keep track of.
titanic <- na.omit(titanic)

#####Linearity-Logit Assumption Check#####
#Breaking down data into train and test split
titanic <- titanic[,c(1,2,4,5,6,7,8)]
set.seed(2)
vec <- sample(c(1:nrow(titanic)), (nrow(titanic)/3),replace = FALSE)
test <- titanic[vec,]
train <- titanic[-vec,]
#Now we can actually generate the model
model <- glm(Survived ~., data = train, 
             family = binomial)

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
#Conclusion: we may need to apply a spline to the "age" predictor.  The "fare" predictor does not appear to need a spline.  Therefore, we will proceed
#with a model that DOES include a spline on the "age" predictor and one that does NOT include a spline on the "age" predictor.  We will then compare their
#respective performances against cross-validation and see if the inclusion in a spline is justified.
#####Assumption of No Influential Outliers#####
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


#Optimizing the number of degrees of freedom via cross-validation; we suspect there would be up to
# 3 regions (df = 2) for age based on graphics from exploratory phase; we want to allow for the possibility that each area of the domain might need it's own piece
#of the piece-wise polynomial
#We'll add one in the interest of exploring additional possibilities

#Histrogram to provide some additional intuition about how many degrees of freedom may be appropriate
titanic %>% ggplot(aes(x = Age))+
  geom_density()+
  xlab("Age")+
  ylab("Density")+
  ggtitle("Estimating Initial Degrees of Freedom: Age")


#Here we use cross validation to find out what the ideal number of degrees of freedom for the Age spline is.  We randomly select 1/3 of the data to test and 2/3 to train
#For each iteration, each number of degrees of freedom (1-4) will be used to create a model and accuracy will be calculated.  We define accuracy as simply a 
#result that matches a prediction.  Inaccuracies are results that do not match a prediction; we do not distinguish between false positives and false negatives
#for these purposes.  The degree of freedom that is most frequently the most accurate will then be adopted by the model.
#setting seed and shuffling data
set.seed(1)
rows <- sample(nrow(titanic))
shuffled <- titanic[rows,]

#Then we split our data set for 3-fold cross validation
t <- nrow(shuffled)/5
df1 <- shuffled[1:t,]
df2 <- shuffled[(t+1):(2*t),]
df3 <- shuffled[((2*t)+1):(3*t),]
df4 <- shuffled[((3*t)+1):(4*t),]
df5 <- shuffled[((4*t)+1):(5*t),]

#Beginning first fold
train1 <- bind_rows(df2,df3, df4, df5)
test1 <- df1

test1 <- test1 %>% mutate(survived_numeric = as.numeric(Survived)-1)

accuracy_data_1 <- data.frame()

for(i in 1:4)
{
  

model_1 <- glm(Survived ~ +SibSp + ns(Age, df = i) + Pclass + Parch + Fare,
             data = train1,
             family = binomial)
model_1 <- step(model_1, trace = FALSE)

df_1 <- test1 %>% mutate(predicted_percent_survival = predict(newdata = test1, model_1, type = "response"))
df_1$predicted_percent_survival <- round(df_1$predicted_percent_survival)
df_1 <- df_1 %>% mutate(diff = abs(survived_numeric-predicted_percent_survival))
accuracy_data_1[i,1] <- 1-(sum(df_1$diff)/nrow(df_1))
accuracy_data_1[i,2] <- i
df_1 <- NA
}
colnames(accuracy_data_1)[1] <- "Accuracy1"

#Beginning second fold
train2 <- bind_rows(df1,df3, df4, df5)
test2 <- df2
test2 <- test2 %>% mutate(survived_numeric = as.numeric(Survived)-1)


accuracy_data_2 <- data.frame()

for(i in 1:4)
{
  
  
  model_2 <- glm(Survived ~ +SibSp + ns(Age, df = i) + Pclass + Parch + Fare,
                 data = train2,
                 family = binomial)
  model_2 <- step(model_2, trace = FALSE)
  
  df_2 <- test2 %>% mutate(predicted_percent_survival = predict(newdata = test2, model_2, type = "response"))
  df_2$predicted_percent_survival <- round(df_2$predicted_percent_survival)
  df_2 <- df_2 %>% mutate(diff = abs(survived_numeric-predicted_percent_survival))
  accuracy_data_2[i,1] <- 1-(sum(df_2$diff)/nrow(df_2))
  accuracy_data_2[i,2] <- i
  df_2 <- NA
}
colnames(accuracy_data_2)[1] <- "Accuracy2"

#Beginning third fold
train3 <- bind_rows(df1,df2, df4, df5)
test3 <- df3
test3 <- test3 %>% mutate(survived_numeric = as.numeric(Survived)-1)


accuracy_data_3 <- data.frame()

for(i in 1:4)
{
  
  
  model_3 <- glm(Survived ~ +SibSp + ns(Age, df = i) + Pclass + Parch + Fare,
                 data = train3,
                 family = binomial)
  model_3 <- step(model_3, trace = FALSE)
  
  df_3 <- test3 %>% mutate(predicted_percent_survival = predict(newdata = test3, model_3, type = "response"))
  df_3$predicted_percent_survival <- round(df_3$predicted_percent_survival)
  df_3 <- df_3 %>% mutate(diff = abs(survived_numeric-predicted_percent_survival))
  accuracy_data_3[i,1] <- 1-(sum(df_3$diff)/nrow(df_3))
  accuracy_data_3[i,2] <- i
  df_3 <- NA
}

colnames(accuracy_data_3)[1] <- "Accuracy3"

#Starting fourth fold
train4 <- bind_rows(df1,df2, df3, df4)
test4 <- df4
test4 <- test4 %>% mutate(survived_numeric = as.numeric(Survived)-1)


accuracy_data_4 <- data.frame()

for(i in 1:4)
{
  
  
  model_4 <- glm(Survived ~ +SibSp + ns(Age, df = i) + Pclass + Parch + Fare,
                 data = train4,
                 family = binomial)
  model_4 <- step(model_4, trace = FALSE)
  
  df_4 <- test4 %>% mutate(predicted_percent_survival = predict(newdata = test4, model_4, type = "response"))
  df_4$predicted_percent_survival <- round(df_4$predicted_percent_survival)
  df_4 <- df_4 %>% mutate(diff = abs(survived_numeric-predicted_percent_survival))
  accuracy_data_4[i,1] <- 1-(sum(df_4$diff)/nrow(df_4))
  accuracy_data_4[i,2] <- i
  df_4 <- NA
}

colnames(accuracy_data_4)[1] <- "Accuracy4"

#Starting fifth fold

train5 <- bind_rows(df1,df2, df3, df4)
test5 <- df5
test5 <- test5 %>% mutate(survived_numeric = as.numeric(Survived)-1)


accuracy_data_5 <- data.frame()

for(i in 1:4)
{
  
  
  model_5 <- glm(Survived ~ +SibSp + ns(Age, df = i) + Pclass + Parch + Fare,
                 data = train5,
                 family = binomial)
  model_5 <- step(model_5, trace = FALSE)

  df_5 <- test5 %>% mutate(predicted_percent_survival = predict(newdata = test5, model_5, type = "response"))
  df_5$predicted_percent_survival <- round(df_5$predicted_percent_survival)
  df_5 <- df_5 %>% mutate(diff = abs(survived_numeric-predicted_percent_survival))
  accuracy_data_5[i,1] <- 1-(sum(df_5$diff)/nrow(df_5))
  accuracy_data_5[i,2] <- i
  df_5 <- NA
}

colnames(accuracy_data_5)[1] <- "Accuracy5"

max1<-accuracy_data_1 %>% filter(Accuracy1==max(Accuracy1))
max2<-accuracy_data_2 %>% filter(Accuracy2==max(Accuracy2))
max3<-accuracy_data_3 %>% filter(Accuracy3==max(Accuracy3))
max4<-accuracy_data_4 %>% filter(Accuracy4==max(Accuracy4))
max5<-accuracy_data_5 %>% filter(Accuracy5==max(Accuracy5))

max1
max2
max3
max4
max5
#Notice that 3 degrees of freedom consistently provides the most accurate results; this is what we will use going forward


#####Addressing Colinearity with backward selection#####
#Now that we know how many degrees of freedom we want to use in the spline model, we can continue building our models, one with a spline on
#the "age" predictor, and one without.  We now address collinearity possibilities with
#a backward selection process similar to what we used in linear modeling. (note, backward selection is the default for the "step()" function.)
model_spline <- glm(Survived ~ SibSp + ns(Age, df = 3) + Pclass + Parch + Fare + sex_binary,
             data = titanic,
             family = binomial)
model_spline <- step(model_spline, trace = FALSE)
summary(model_spline)
#Spline model features include; Siblings; Age; Class; Sex

model_no_spline <- glm(Survived ~ SibSp + Age + Pclass + Parch + Fare + sex_binary,
                       data = titanic,
                       family = binomial)
model_no_spline <- step(model_no_spline, trace = FALSE)
summary(model_no_spline)

#No-spline model features includel; siblings; Age; Class; Sex
#########Cross-Validation/Comparing Models###################
AIC(model_no_spline)
AIC(model_spline)

#cross validation to get additional information about how our models perform
#first we shuffle our rows randomly
set.seed(3)
rows <- sample(nrow(titanic))
shuffled <- titanic[rows,]

#Then we split our data set for 3-fold cross validation
t <- nrow(shuffled)/5
df1 <- shuffled[1:t,]
df2 <- shuffled[(t+1):(2*t),]
df3 <- shuffled[((2*t)+1):(3*t),]
df4 <- shuffled[((3*t)+1):(4*t),]
df5 <- shuffled[((4*t)+1):(5*t),]

#Beginning first fold
train1 <- bind_rows(df2,df3, df4, df5)
test1 <- df1

model_spline_1 <- glm(Survived ~SibSp + ns(Age, df = 3) + Pclass + sex_binary,
              data = train1, 
             family = binomial)
test1 <- test1 %>% mutate(predicted_percent_survival_no_spline = predict(model_spline_1, newdata = test1, type = "response"))

model_no_spline_1 <- glm(Survived ~SibSp + Age + Pclass + sex_binary,
                      data = train1, 
                      family = binomial)
test1 <- test1 %>% mutate(predicted_percent_survival_spline = predict(model_no_spline_1, newdata = test1, type = "response"))

test1$predicted_percent_survival_no_spline <- round(test1$predicted_percent_survival_no_spline)
test1$predicted_percent_survival_spline <- round(test1$predicted_percent_survival_spline)

test1 <- test1 %>% mutate(survived_numeric = as.numeric(Survived)-1)

test1 <- test1 %>% mutate(diff_no_spline = abs(survived_numeric-predicted_percent_survival_no_spline))
test1 <- test1 %>% mutate(diff_spline = abs(survived_numeric-predicted_percent_survival_spline))



accuracy1_no_spline <- 1-(sum(test1$diff_no_spline)/nrow(test1))
accuracy1_spline <- 1-(sum(test1$diff_spline)/nrow(test1))


#Beginning second fold
train2 <- bind_rows(df1,df3, df4, df5)
test2 <- df2

model_spline_2 <- glm(Survived ~SibSp + ns(Age, df = 3) + Pclass + sex_binary,
                      data = train2, 
                      family = binomial)
test2 <- test2 %>% mutate(predicted_percent_survival_no_spline = predict(model_spline_2, newdata = test2, type = "response"))

model_no_spline_2 <- glm(Survived ~SibSp + Age + Pclass + sex_binary,
                         data = train2, 
                         family = binomial)
test2 <- test2 %>% mutate(predicted_percent_survival_spline = predict(model_no_spline_2, newdata = test2, type = "response"))

test2$predicted_percent_survival_no_spline <- round(test2$predicted_percent_survival_no_spline)
test2$predicted_percent_survival_spline <- round(test2$predicted_percent_survival_spline)

test2 <- test2 %>% mutate(survived_numeric = as.numeric(Survived)-1)

test2 <- test2 %>% mutate(diff_no_spline = abs(survived_numeric-predicted_percent_survival_no_spline))
test2 <- test2 %>% mutate(diff_spline = abs(survived_numeric-predicted_percent_survival_spline))



accuracy2_no_spline <- 1-(sum(test2$diff_no_spline)/nrow(test2))
accuracy2_spline <- 1-(sum(test2$diff_spline)/nrow(test2))
#beginning third fold
train3 <- bind_rows(df1,df2, df4, df5)
test3 <- df3

model_spline_3 <- glm(Survived ~SibSp + ns(Age, df = 3) + Pclass + sex_binary,
                      data = train3, 
                      family = binomial)
test3 <- test3 %>% mutate(predicted_percent_survival_no_spline = predict(model_spline_3, newdata = test3, type = "response"))

model_no_spline_3 <- glm(Survived ~SibSp + Age + Pclass + sex_binary,
                         data = train3, 
                         family = binomial)
test3 <- test3 %>% mutate(predicted_percent_survival_spline = predict(model_no_spline_3, newdata = test3, type = "response"))

test3$predicted_percent_survival_no_spline <- round(test3$predicted_percent_survival_no_spline)
test3$predicted_percent_survival_spline <- round(test3$predicted_percent_survival_spline)

test3 <- test3 %>% mutate(survived_numeric = as.numeric(Survived)-1)

test3 <- test3 %>% mutate(diff_no_spline = abs(survived_numeric-predicted_percent_survival_no_spline))
test3 <- test3 %>% mutate(diff_spline = abs(survived_numeric-predicted_percent_survival_spline))



accuracy3_no_spline <- 1-(sum(test3$diff_no_spline)/nrow(test3))
accuracy3_spline <- 1-(sum(test3$diff_spline)/nrow(test3))

#Starting fourth fold

train4 <- bind_rows(df1,df2, df3, df5)
test4 <- df4

model_spline_4 <- glm(Survived ~ SibSp + ns(Age, df = 3) + Pclass + sex_binary,
                      data = train4, 
                      family = binomial)
test4 <- test4 %>% mutate(predicted_percent_survival_no_spline = predict(model_spline_4, newdata = test4, type = "response"))

model_no_spline_4 <- glm(Survived ~SibSp + Age + Pclass + sex_binary,
                         data = train4, 
                         family = binomial)
test4 <- test4 %>% mutate(predicted_percent_survival_spline = predict(model_no_spline_4, newdata = test4, type = "response"))

test4$predicted_percent_survival_no_spline <- round(test4$predicted_percent_survival_no_spline)
test4$predicted_percent_survival_spline <- round(test4$predicted_percent_survival_spline)

test4 <- test4 %>% mutate(survived_numeric = as.numeric(Survived)-1)

test4 <- test4 %>% mutate(diff_no_spline = abs(survived_numeric-predicted_percent_survival_no_spline))
test4 <- test4 %>% mutate(diff_spline = abs(survived_numeric-predicted_percent_survival_spline))



accuracy4_no_spline <- 1-(sum(test4$diff_no_spline)/nrow(test4))
accuracy4_spline <- 1-(sum(test4$diff_spline)/nrow(test4))

#starting fifth fold

train5 <- bind_rows(df1,df2, df3, df4)
test5 <- df5

model_spline_5 <- glm(Survived ~ SibSp + ns(Age, df = 3) + Pclass + sex_binary,
                      data = train5, 
                      family = binomial)
test5 <- test5 %>% mutate(predicted_percent_survival_no_spline = predict(model_spline_5, newdata = test5, type = "response"))

model_no_spline_5 <- glm(Survived ~SibSp + Age + Pclass + sex_binary,
                         data = train5, 
                         family = binomial)
test5 <- test5 %>% mutate(predicted_percent_survival_spline = predict(model_no_spline_5, newdata = test5, type = "response"))

test5$predicted_percent_survival_no_spline <- round(test5$predicted_percent_survival_no_spline)
test5$predicted_percent_survival_spline <- round(test5$predicted_percent_survival_spline)

test5 <- test5 %>% mutate(survived_numeric = as.numeric(Survived)-1)

test5 <- test5 %>% mutate(diff_no_spline = abs(survived_numeric-predicted_percent_survival_no_spline))
test5 <- test5 %>% mutate(diff_spline = abs(survived_numeric-predicted_percent_survival_spline))



accuracy5_no_spline <- 1-(sum(test5$diff_no_spline)/nrow(test5))
accuracy5_spline <- 1-(sum(test5$diff_spline)/nrow(test5))

#Evaluation
#in general, we can feel confident that our model predicts who would survive or not survive the Titanic wreck with ~80% accuracy
mean_accuracy_no_spline <- (accuracy1_no_spline + accuracy2_no_spline + accuracy3_no_spline + accuracy4_no_spline + accuracy5_no_spline)/5
mean_accuracy_no_spline

mean_accuracy_spline <- (accuracy1_spline + accuracy2_spline + accuracy3_spline + accuracy5_spline + accuracy5_spline)/5
mean_accuracy_spline

##Using AIC as a means of comparing models

#Notice both models are used to predict the same outcome on the same data set
#Since we have a sufficiently large sample size and can show that the ratio of data points to predictors is greater than 40 we can use AIC as a 
#means of comparing models

nrow(titanic)/4 > 40
model_no_spline <- glm(Survived ~SibSp + Age + Pclass + sex_binary,
                       data = titanic, 
                       family = binomial)
model_spline <- glm(Survived ~SibSp + ns(Age,3) + Pclass + sex_binary,
                       data = titanic, 
                       family = binomial)


AIC(model_spline)
exp((AIC(model_spline) - AIC(model_no_spline))/2)
#The model without the spline is only 0.0141 times as likely as the model with the spline to be the model that minimizes out of sample error
#the AIC criteria clearly favors the use of a spline


#The models demonstrate very similar performance during cross validation; and since the AIC score favors the use of the spline on the "age" predictor, that
#is the model we will use going forward
#####Model Inteprretation#####

model_spline <- glm(Survived ~SibSp + ns(Age,3) + Pclass + sex_binary,
                         data = titanic, 
                         family = binomial)
summary(model_spline)

#Recall "sex_binary is 1 for male, 0 for female
#The log odds decrease as the number of siblings a passenger has increases
#The log odds of survival are lower for second class passengers as opposed to first class(the reference class) and the log odds of survival are MUCH
#lower for passengers in 3rd class than first class
#The log odds of survival are much lower for males than for females

#Notice that the results related to class and sex especially align with our analysis from the "Additional Data Visualizations" section
#Also notice that the "fare" predictor was eliminated during the backward selection process for both models, this is likely because
#that information is carried in the Pclass predictor.
#Now we can graph predicted log odds versus age

model_age <- glm(Survived ~ ns(Age,3),
                 data = titanic,
                 family = binomial)

p1 <- predict(model_age, type = "response")

p <- predict(model_spline, type = "response")
t <- titanic$Age
s <- ns(titanic$Age, 3)
s1 <- s[,1]
s2 <- s[,2]
s3 <- s[,3]

dfAge <- as.data.frame(cbind(t,p,p1,s1,s2,s3))

colnames(dfAge)[1] <- "Age"
colnames(dfAge)[2] <- "Predicted"
colnames(dfAge)[3] <- "Predicted_Age"
colnames(dfAge)[4] <- "S1"
colnames(dfAge)[5] <- "S2"
colnames(dfAge)[6] <- "S3"

dfAge %>% ggplot(aes(x = Age))+
  geom_point(aes(y = LO_Predicted))+
  geom_smooth(aes(y = LO_Predicted))+
  geom_point(aes(y = LO_Predicted_Age), color = "orange")+
  xlab("Age")+
  ylab("Predicted Log Odds of Survival")+
  ggtitle("Predicted Log Odds of Survival")

#In the above plot, the predicted log odds from the final model are in black (with regression in blue), the predicted log odds from a model using age with cubic
#splines as the only feature are in orange.  In both the cases of the regression and the model with only the age spline, we see that there is an 
#age range between approximately 20-45 where an increase in age actually increases the log odds of survival.  We also note that both the regression curve
#and the predictions from the age-only model are only positive for ages below 10 years old, indicating that the under 10 age group is the only age group
#more likely to survive than not survive.  Recall that this aligns with analysis done in the "Additional Data Visualizations" file.

dfAge %>% ggplot(aes(x = Age))+
  geom_line(aes(y = S1), color = "purple")+
  geom_line(aes(y = s2), color = "blue")+
  geom_line(aes(y = s3), color = "red")+
  ylab("Spline Values")+
  xlab("Age")+
  ggtitle("Age Spline Basis Functions")
#Finally, we show the basis functions for our age spline.

