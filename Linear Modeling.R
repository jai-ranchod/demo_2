library("class") #needed for knn
library("tidyverse")
library("plotly")
library("forecast")
library("reldist")
library("corrplot")
library("ineq")
library("gglorenz")
library("lubridate")
library("BSDA")
library("car")
library("pedometrics")
library("caret")
library("gtools")
library("readxl")
library("stringr")
library("pdftools")
library("dplyr")
library("tidyverse")
library("scales")
library(readxl)
##################Lung Capacity Data Set#########################
#To create our linear model, we will begin with the "Lung Capacity" data set found in the Marin Lecture Series on statistics.

Lung_Capacity <- read_excel("~/MBA Classes/Spring Semester 2020/Business Analytics/Data Sets/Lung_Capacity.xlsx")
head(Lung_Capacity)

#Here we will be predicting lung capacity based on age, sex, smoking status, and whether or not the subject was a Caesarean birth or not.
#First, we create a correlation matrix to show the relationships between the predictors.  Note that the sex predictor is 1 for "male" and 0 for "female".
#Likewise, "1" indicate a smoker in the smoker column, and "1" represents an individual that did have a Caesarean birth


Matrix <- cor(Lung_Capacity)
corrplot(Matrix)
View(Matrix)

#Here we see that the height and age predictors are positively correlated with each other. In order to eliminate collinearity concerns, we can use the stepVIF() function in R.
#This function uses "Variance Inflation Factors" to determine which predictors have influence that is inflated by the presence of other predictors in the model.
#The stepVIF() function iteratively removes predictors with a VIF above a certain threshold until only sufficiently independent predictors remain.
#First we coerce our categorical variables to factors, then we create an initial model, then we apply the stepVIF() function.


Capacity <- Lung_Capacity
Capacity$Smoker_YN <- as.factor(Capacity$Smoker_YN)
Capacity$Male_YN <- as.factor(Capacity$Male_YN)
Capacity$Caesarean <- as.factor(Capacity$Caesarean_YN)

Lung_Cap <- Capacity$LungCap
Age <- Capacity$Age
Height <- Capacity$Height
Smoker <- Capacity$Smoker_YN
Male_YN <- Capacity$Male_YN
Caesarean_YN <- Capacity$Caesarean_YN

Model <- lm(Lung_Cap ~ Age + Height + Smoker + Male_YN + Caesarean_YN)
stepVIF(Model)

#All of our predictors meet our threshold and can stay in the model for now.  Next we see what the model looks like.

summary(Model)

#Here we see that we are describing approximately 85% of the variation in lung capacity with our predictors.  We further see that we have a highly statistically significant F-statistic.
#Finally, we note that all of our predictors have a p-value less than 0.05, indicating they are all statistically significant.  If this were not the case, we would begin the "backward selection" process by removing the predictor with the highest p-value and re-running the model until all predictors were below the significance threshold.
#Unsurprisingly we note that being a smoker reduces lung capacity.  We note as well that age increases lung capacity which may seem counterintuitive, until we see that the maximum age in this dataset is 19.

min(Lung_Capacity$Age)
max(Lung_Capacity$Age)

#We finally must make sure we are meeting the assumptions of linear modeling.  We do not know anything about the data collection process, so we cannot be sure that the samples are genuinely independent.
#However, we will assume for now that they are.  Next, we can generate diagnostic plots with the simple plot() function.

par(mfrow = c(2,2))
plot(Model)
par(mfrow = c(1,1))

#The residuals plot in the upper left-hand corner shows us that we are meeting our linearity and homoscedasticity assumptions.
#The qq plot in the upper right hand corner shows us that we are meeting our normality assumption as well.
#In general, we can fell quite good that we have a useful linear model for predicting the lung capacity of individuals between the ages of 3 and 19.
##################Cars data Set######################
#We can also build a model with the "mtcars" dataset pre-built into R.
data(mtcars)
#mpg - Miles per Gallon
#cyl - number of cylinders
#disp - engine displacement; combined volume of cylinders
#hp - horsepower
#drat - rear axle ratio
#wt - weight
#qsec - quarter mile time
#vs - is engine "V" shaped (1) of straight (0)
#am - automatic transmission (0) or manual transmission (1) (this dataset is from 1974, so manual transmission could very well have a positive impact on feul economy)
#gear - number of forward gears
#carb - number of carburetors


#In this model, we attempt to model gas mileage.  As before, we begin by showing a graphic for the correlation matrix of the predictors.
mtcars_test <- mtcars[-1]
Matrix <- cor(mtcars_test)
corrplot(Matrix)
#View(Matrix) - lets you see the actual numbers of the correlation matrix

#Here we see results that match our Principal Component Analysis, substituting the code:

#mtcars_test <- mtcars[-1]
#Matrix <- cor(mtcars_test)
#corrplot(Matrix)


#lets you see the correlation matrix graphic with the mpg vector



#We see a variety of positive and negative corrleations appearing here, but again, we ultimately use the stepVIF() function to determine collinearity.
Cars <- mtcars
#Before creating our initial model we convert categorical predictors to data type "factor"
Cars$vs <- as.factor(Cars$vs)
Cars$am <- as.factor(Cars$am)

Mileage <- Cars$mpg
Cylinders <- Cars$cyl
Displacement <- Cars$disp
Horsepower <- Cars$hp
Rear_Axle_Ratio <- Cars$drat
Weight <- Cars$wt
Acceleration <- Cars$qsec
Engine <- Cars$vs
Transmission <- Cars$am
Gears <- Cars$gear
Carburetors <- Cars$carb

Cars_Model <- lm(Mileage ~ Cylinders + Displacement + Horsepower + Rear_Axle_Ratio + Weight + Acceleration + Engine + Transmission + Gears + Carburetors)

stepVIF(Cars_Model)

#We do see featuers eliminated by the stepVIF() function in this case.  The below model contains the features remaining. Recall that the default threshold for the stepVIF() function is 10.
#We can now be confident in the independence of our predictors moving forward.

Cars_Model <- lm(Mileage ~ Horsepower + Rear_Axle_Ratio + Weight + Acceleration + Engine + Transmission + Gears + Carburetors)
summary(Cars_Model)

#Here we see that there are predictors not meeting our p <= 0.05 significance threshold.  We therefore use a "backward selection" process to eliminate them.
#We remove the least significant predictor, re-run the model and repeat until all predictors are significant.  Below is the resulting model.

Cars_Model_2 <- step(Cars_Model)
summary(Cars_Model_2)

#Again, we examime the diagnostic plots to check the assumptions of linearity.
par(mfrow = c(2,2))
plot(Cars_Model_2)
par(mfrow = c(1,1))

#We again embrace the assumption of independent observations, as the fuel efficiency of one car would not affect that of another.
#We notice that the qq plot in the upper right hand corner appears to confirm the normality assumption of linear modeling.
#The residuals plot in the upper left, however, places some doubt on our linearity and homoscedasticity assumptions.
#This could mean more data points are needed, or that fuel efficiency is not conducive to a linear model.  For now, we can say that we should be skeptical
#of using a linear model to estimate fuel efficiency. 

#I am including this in the demontration to show my ability to evaluate a situation for whether or 
#not a linear model may be appropriate

