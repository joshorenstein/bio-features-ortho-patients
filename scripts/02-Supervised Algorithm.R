#Load R packages

Packages <- c("here", "caret", "rpartScore", "tidyverse", "recipes","C50","yardstick","e1071",
              "rattle","rpart.plot","reshape")

lapply(Packages, library, character.only = TRUE)

#Load the data
#df <- read_csv("data/column_3C_weka.csv")
df <- summary

#Just focus on Normal vs Abnormal as the dependent variable
df <- 
  df %>% 
  mutate(class=if_else(class=="Normal","Normal","Abnormal")) %>% 
  filter(degree_spondylolisthesis<=360) #remove crazy outlier
#names(df)
#str(df$cluster)
df$class <- as.factor(df$class)
df$cluster <- as.numeric(df$cluster)
#do some exploratory analysis
tidy <- df %>% 
  gather(key = "type","degrees",1:6)

bp <- ggplot(tidy, aes(x=class, y=degrees), col=class) +
  geom_boxplot()

b_plot <- bp + facet_grid(. ~ type)
b_plot
ggsave("charts/box_plot.pdf",width=10,height=4)
??ggsave
names(df)
df <- df %>% select(class,cluster,everything())
#Recursive Feature Elimintation - Accuracy and Kappa at its greatest with all six variables
control <- rfeControl(functions=rfFuncs, method="cv", number=2)
results <- rfe(df[, 2:8], df[[1]], sizes=c(1:7), rfeControl=control)
results

#keep datapoints that will be used in model
# Split the data into training and testing sets
set.seed(5818)
in_train <- createDataPartition(df$class, p=0.75, list = FALSE) #stratify by playoff
training <- df[in_train,]
testing <- df[-in_train,]

training %>% group_by(class) %>% count()
names(summary)
#tuning controls and then model fits
ctrl <- trainControl(method = "repeatedcv",number=10,repeats=3,sampling="up",
                     summaryFunction = defaultSummary,
                     savePredictions = "final")

cart_model <- train(class~pelvic_incidence+pelvic_tilt+lumbar_lordosis_angle+
                      sacral_slope+pelvic_radius+degree_spondylolisthesis+(1|cluster), method = "rpart2", #regression tree
                    data = training,
                    trControl = ctrl)

rf_model  <- train(class~pelvic_incidence+pelvic_tilt+lumbar_lordosis_angle+
                     sacral_slope+pelvic_radius+degree_spondylolisthesis+(1|cluster), method = "rf", #random forest
                   data = training,
                   trControl = ctrl)

gbm_model <- train(class~pelvic_incidence+pelvic_tilt+lumbar_lordosis_angle+
                     sacral_slope+pelvic_radius+degree_spondylolisthesis+(1|cluster), method = "gbm", #gradient boost
                   data = training,
                   trControl = ctrl)

svm_model <- train(class~pelvic_incidence+pelvic_tilt+lumbar_lordosis_angle+
                     sacral_slope+pelvic_radius+degree_spondylolisthesis+(1|cluster), method = "svmLinear", varImp.train=TRUE, #support vector
                   data = training,
                   trControl = ctrl)

#svm with no fixed effect
svm_model_no <- train(class~pelvic_incidence+pelvic_tilt+lumbar_lordosis_angle+
                     sacral_slope+pelvic_radius+degree_spondylolisthesis, method = "svmLinear", varImp.train=TRUE, #support vector
                   data = training,
                   trControl = ctrl)

c50_model <- train(class~pelvic_incidence+pelvic_tilt+lumbar_lordosis_angle+
                     sacral_slope+pelvic_radius+degree_spondylolisthesis+(1|cluster), method = "C5.0", varImp.train=TRUE, #C5.0 tree
                   data = training,
                   trControl = ctrl)

nnet_model <- train(class~pelvic_incidence+pelvic_tilt+lumbar_lordosis_angle+
                      sacral_slope+pelvic_radius+degree_spondylolisthesis+(1|cluster), method = "nnet", varImp.train=TRUE, #neural net
                    data = training,
                    trControl = ctrl)

# View some preliminary results 
summary(cart_model)
summary(gbm_model) #gbm is nice because it gives variable importance
summary(svm_model)
summary(c50_model)
summary(rf_model)
summary(nnet_model)

fancyRpartPlot(cart_model$finalModel)	

dev.copy(pdf,'charts/regression-tree-model.pdf')
dev.off()

plot(nnet_model$finalModel) 
dev.copy(pdf,'charts/neural-net-model.pdf')
dev.off()
# Create new columns with results from training
training_results <- training %>%
  mutate(Decision_Tree = predict(cart_model, training),
         Random_Forest = predict(rf_model, training),
         Gradient_Boost = predict(gbm_model, training),
         Support_Vector = predict(svm_model, training),
         C50 = predict(c50_model,training),
         Neural_Net = predict(nnet_model,training),
         Support_Vector_No = predict(svm_model_no,training))

# Evaluate the performance
g <- getTrainPerf(cart_model)
h <- getTrainPerf(rf_model)
i <- getTrainPerf(gbm_model)
j <- getTrainPerf(svm_model)
k <- getTrainPerf(c50_model)
l <- getTrainPerf(nnet_model)
k <- getTrainPerf(svm_model_no)
#look at all the training results together
train_perf <- bind_rows(g,h,i,j,k,l,k)
train_perf %>% arrange(desc(TrainAccuracy))  #results fairly similiar

# Create the new columns in the test set and see how it performs
testing_results <- testing %>%
  mutate(Decision_Tree = predict(cart_model, testing),
         Random_Forest = predict(rf_model, testing),
         Gradient_Boost = predict(gbm_model, testing),
         Support_Vector = predict(svm_model, testing),
         C50 = predict(c50_model,testing),
         Neural_Net = predict(nnet_model,testing),
         Support_Vector_No = predict(svm_model_no,testing))

#create some metrics to evaluate all results
f <- function(data,algorithm){
  metrics(data,truth=class,estimate=algorithm) %>% 
    mutate(model=algorithm)}

m <- f(training_results,"Decision_Tree")
n <- f(training_results,"Gradient_Boost")
o <- f(training_results,"Support_Vector")
p <- f(training_results,"C50")
q <- f(training_results,"Random_Forest")
r <- f(training_results,"Neural_Net")
s <- f(training_results,"Support_Vector_No")
m1 <- f(testing_results,"Decision_Tree")
n1 <- f(testing_results,"Gradient_Boost")
o1 <- f(testing_results,"Support_Vector")
p1 <- f(testing_results,"C50")
q1 <- f(testing_results,"Random_Forest")
r1 <- f(testing_results,"Neural_Net")
s1 <- f(testing_results,"Support_Vector_No")
#see the train/test kappa (accuracy vs randomness)
models <- bind_rows(m,n,o,p,q,r,s) %>% filter(.metric=="accuracy")
m <- as.data.frame(models) %>% mutate(data="training")
models1 <- bind_rows(m1,n1,o1,p1,q1,r1,s1) %>% filter(.metric=="accuracy")
m1 <- as.data.frame(models1) %>% mutate(data="test") %>% bind_rows(m) %>% 
  select(data,model,.estimate) %>% 
  mutate(.estimate=.estimate*100) %>% 
  arrange(model,desc(data)) %>% 
  group_by(model) %>% 
  spread(data,.estimate) %>% 
  mutate(diff=test-training) %>% 
  arrange(desc(test))
m1 %>% write_csv("results/all_model_performance.csv")
 
#Random Forest & Gradient Boost likely overfit, Support Vector only model that performed better on test than train 

# Take a look at a few confusion matrices
#---
confusionMatrix(predict(cart_model,testing),
                testing$class) #abnormal predictions are good, normal predictions are coin flip

#highest balanced accuracy
a <- confusionMatrix(predict(svm_model,testing),
                testing$class) #normal predictions much improved from cart, abnormal predictions really strong

b <- as.table(a)
c <- as.data.frame(a$byClass)

b %>% write.table("results/predictions.csv")
c %>% write.table("results/model_accuracy.csv")

#compare to svm with no fixed effect
a1 <- confusionMatrix(predict(svm_model_no,testing),
                     testing$class) #normal predictions much improved from cart, abnormal predictions really strong


confusionMatrix(predict(nnet_model,testing),
                testing$class) #similar results to svm

#Move forward with the neural net due to its balaanced accuracy in test/train 
results <- testing_results %>% 
  select(-c(Decision_Tree,Random_Forest,Gradient_Boost,C50,Support_Vector,Support_Vector_No)) %>% 
  dplyr::rename(prediction=Neural_Net)

names(results)
results %>% 
  gather(key = "type","degrees",3:8)
names(results)
medians <- results %>% 
  gather("variable", "value",3:8) %>% 
  group_by(cluster,prediction,variable) %>% 
  summarize(lower = round(quantile(value, probs = .25),0),
            median = round(quantile(value, probs = .5),0),
            upper = round(quantile(value, probs = .75),0)) %>% 
  arrange(variable) #%>% 
  # gather("quartile","value",3:5) 

medians %>% write_csv('results/results.csv')

#estimate weights of the support vector model
fit2 <- svm(class ~ ., data = training)
w <- t(fit2$coefs) %*% fit2$SV                 # weight vectors
w <- apply(w, 2, function(v){sqrt(sum(v^2))})  # weight
w <- sort(w, decreasing = T)
w <- as.data.frame(w)
w <- tibble::rownames_to_column(w, "Type")
w <- w %>% mutate(w=round(w,1)) 
w
w1 <- w %>% 
  mutate(w, variable_importance = round(w / sum(w),2)*100) %>% 
  select(-w)
## set the levels in order we want
library(ggthemes)
## plot
q <- ggplot(w1, aes(y = reorder(Type, variable_importance), x = variable_importance),fill=black) + theme_minimal() +
  geom_bar(stat="identity", position = "dodge") +
  labs(x="Variable Importance",y="Feature",title="Variable Importance",
       subtitle="Biomechanical features impacting presence of hernia in orthopedic patients")
q

ggsave("results/variable_importance.pdf",width=8,height=4)

