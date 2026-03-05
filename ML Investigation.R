
################ Load Libraries and Data ###################

library(dplyr)
library(tidyr)
library(umap)
library(reticulate)
library(MSEtool)
library(ggplot2)
library(RColorBrewer)
library(tibble)
library(stringr)
library(ggnewscale)
library(vegan)
library(ggthemes)
library(FNN)
library(factoextra)
library(caret)
library(e1071)
library(caTools)
library(xgboost)
library(pROC)
library(randomForest)
library(keras3)
library(shapr)
library(iml)
library(shapviz)
library(fastshap)


# Load data
bcancer <- read.csv("\\breast+cancer+wisconsin+diagnostic\\wdbc.data", header = F)
# Add column names
colnames(bcancer) <- c("ID", "Diagnosis", "Radius1", "Texture1", "Perimeter1", "Area1", "Smoothness1", "Compactness1",
                       "Concavity1", "ConcavePoints1", "Symmetry1", "FractalDimension1",
                       "Radius2", "Texture2", "Perimeter2", "Area2", "Smoothness2", "Compactness2","Concavity2", 
                       "ConcavePoints2", "Symmetry2", "FractalDimension2",
                       "Radius3", "Texture3", "Perimeter3", "Area3", "Smoothness3", "Compactness3",
                       "Concavity3", "ConcavePoints3", "Symmetry3", "FractalDimension3")
# Extract subset of just numerical features
bcancer_features <- bcancer[3:32]


############ Log-transform B-Cancer data ############ 

# Log-transform all numeric features
log_bcancer_features <- bcancer_features %>% 
  mutate(across(where(is.numeric), log))
#Add columns for ID and Diagnosis features
log_bcancer <- cbind(log_bcancer_features, "Diagnosis" = bcancer$Diagnosis)
log_bcancer <- cbind("ID" = bcancer$ID, log_bcancer)

# Convert data to long format and view distributions in histograms
columns <- colnames(bcancer[3:32])
bcancer_long <- bcancer %>% pivot_longer(cols = columns,
                                         names_to = "metric",
                                         values_to = "measurement")

ggplot(bcancer_long) + 
  geom_histogram(aes(x = measurement), bins = 50, fill = "turquoise") + 
  coord_cartesian(ylim = c(0,100)) +
  facet_wrap(~metric)


###################### UMAP Analysis ######################

# Remove infinite values
bcancer_umap <- log_bcancer %>%
  filter(if_all(where(is.numeric), is.finite))

# Set up UMAP model and create dataframe of UMAP axes
umap_log_bcancer <- umap(bcancer_umap[2:31], n_components = 2, random_state = 15)
umap_log_bcancer_data = umap_log_bcancer[["layout"]]
umap_log_bcancer_data <- data.frame(umap_log_bcancer_data)

# Add columns of corresponding columns for later visualization
umap_log_bcancer_data <- cbind(umap_log_bcancer_data, "Diagnosis" = bcancer_umap$Diagnosis)
umap_log_bcancer_data <- cbind(umap_log_bcancer_data, "Area1" = bcancer_umap$Area1)
umap_log_bcancer_data <- cbind(umap_log_bcancer_data, "ConcavePoints2" = bcancer_umap$ConcavePoints2)

# View UMAP plot with Diagnosis mapped onto samples
ggplot(umap_log_bcancer_data) + 
  geom_point(aes(x = X1, y = X2, color = Diagnosis), size = 1) + 
  theme_few() + 
  theme(legend.position = "bottom", legend.text = element_text(size = 12)) + 
  labs(x = "UMAP 1", 
       y = "UMAP 2", 
       color = "Diagnosis") + 
  scale_color_discrete(labels = c("Benign", "Malignant")) + 
  guides(color = guide_legend(override.aes = list(size = 3)))

# View UMAP plot with feature mapped onto samples
ggplot(umap_log_bcancer_data) + 
  geom_point(aes(x = X1, y = X2, color = Area1), size = 1) + 
  theme_few() + 
  theme(legend.position = "bottom", legend.text = element_text(size = 12)) + 
  labs(x = "UMAP 1", 
       y = "UMAP 2", 
       color = "Area1") +
  scale_color_viridis_c(option = "magma")


################ PCA Analysis #####################

#  Remove infinite values, create subset of features for PCA, then run it
bcancer_pca <- log_bcancer[apply(log_bcancer_features, 1, function(x) all(is.finite(x))), ]
bcancer_PCA <- princomp(bcancer_pca[2:31])

#  Scree plot
fviz_eig(bcancer_PCA, addlabels = T)
# Loadings plot
fviz_pca_var(bcancer_PCA, col.var = "cos2", gradient.cols = c("black", "red", "pink", "purple", "lightblue", "darkseagreen", "gold"), labelsize = 4, arrowsize = 0.8) +
  labs(title = "PCA of Features for Breast Masses")
# Plot for representation of variables
fviz_cos2(bcancer_PCA, choice = "var", axes = 1:2)

# View samples along 2 biggest PC axes
bcancer_pca$PCA1 <- bcancer_PCA$scores[,1]
bcancer_pca$PCA2 <- bcancer_PCA$scores[,2]
ggplot(data = bcancer_pca) +
  geom_point(aes(x = PCA1, y = PCA2, color = Diagnosis), size = 1) + 
  theme_few() + 
  theme(legend.position = "bottom") +
  labs(x = "PCA 1", 
       y = "PCA 2", 
       color = "Diagnosis") + 
  scale_color_discrete(labels = c("Benign", "Malignant")) + 
  guides(color = guide_legend(override.aes = list(size = 3)))


############################ Random Forest ##################################

# Remove infinite values and add Diagnosis as a factor column to data for RF model
bcancer_rf <- log_bcancer[apply(log_bcancer[2:31], 1, function(x) all(is.finite(x))), ]
# bcancer_rf <- log_bcancer %>%
#   filter(if_all(where(is.numeric), is.finite))
bcancer_rf$Diagnosis <- as.factor(bcancer_rf$Diagnosis)
bcancer_rf$Diagnosis <- as.numeric(bcancer_rf$Diagnosis) - 1
bcancer_rf$Diagnosis <- factor(bcancer_rf$Diagnosis, levels = c(0,1))

# 70/30 split of training and testing data
set.seed(33)
train_index_all <- sample(1:nrow(bcancer_rf), 0.7 * nrow(bcancer_rf))
str(train_index_all)
training_bcancer_all <- bcancer_rf[train_index_all, ]
training_bcancer_all <- training_bcancer_all[, -1]
testing_bcancer_all <- bcancer_rf[-train_index_all, ]
testing_bcancer_all <- testing_bcancer_all[, -1]

# Train the model
rf_model <- randomForest(Diagnosis ~ .,
                         data = training_bcancer_all,
                         ntree = 500,
                         importance = T)

# Generate predictions
rf_predictions <- predict(rf_model, testing_bcancer_all)
rf_predictions <- as.numeric(as.character(rf_predictions))

# View confusion matrix
confusionMatrix(as.factor(rf_predictions), testing_bcancer_all$Diagnosis)

# View rankings of importance for accuracy of predictions
importance(rf_model)
num_var <- ncol(bcancer_rf) - 1
varImpPlot(rf_model, n.var = nrow(rf_model$importance), main = "Variable Importance Plots based on Random Forest")

# Manual F1 score
precision <- cm["1","1"] / sum(cm["1",])
recall    <- cm["1","1"] / sum(cm[,"1"])
(f1_manual <- 2 * precision * recall / (precision + recall))

# Generate ROC object for AUC score and plot
roc_object_rf <- roc(testing_bcancer_all$Diagnosis, rf_predictions)
auc(roc_object_rf)
plot(roc_object_rf,
     col="blue",
     lwd=3,
     print.auc=TRUE,
     legacy.axes=TRUE)
abline(a=0,b=1,lty=2,col="gray")


################### XGBoost Model #######################

# Remove infinite values and add Diagnosis as a factor column to data for XGB model
bcancer_xgb <- log_bcancer[apply(log_bcancer[2:31], 1, function(x) all(is.finite(x))), ]
bcancer_xgb$Diagnosis <- factor(bcancer_xgb$Diagnosis, levels = c("B", "M"))
bcancer_xgb$Diagnosis <- as.numeric(bcancer_xgb$Diagnosis) - 1

# 70/30 split of training and testing data
set.seed(409)
xgb_split <- sample.split(bcancer_xgb$Diagnosis, SplitRatio = 0.7)
training_set <- subset(bcancer_xgb, xgb_split == T)
training_set <- training_set[, -1]
testing_set <- subset(bcancer_xgb, xgb_split == F)
testing_set <- testing_set[, -1]
xgb_train_x <- model.matrix(Diagnosis ~ ., data = training_set)
xgb_test <- model.matrix(Diagnosis ~ ., data = testing_set)
xgb_train_label <- training_set$Diagnosis
final_xgb_train <- xgb.DMatrix(data = xgb_train_x[,-1], label = training_set$Diagnosis)
final_xgb_test <- xgb.DMatrix(data = xgb_test[,-1], label = testing_set$Diagnosis)

# Run XGB model
bcancer_XGB_model <- xgb.train(data = final_xgb_train, 
                               params = list(objective = "binary:logistic",
                                             eta = 0.25), 
                               nrounds = 200)

# Generate predictions, convert them to binary format
xgb_pred <- predict(bcancer_XGB_model, final_xgb_test)
hist(xgb_pred)
xgb_pred <- ifelse(xgb_pred > 0.5, 1, 0)

# View matrix of predictions and actual results
confusionMatrix(as.factor(xgb_pred), as.factor(testing_set$Diagnosis))
(cm <- table(Predicted = as.factor(xgb_pred), Actual = as.factor(testing_set$Diagnosis)))

# Manual F1 value
precision <- cm["0","0"] / sum(cm["0",])
recall    <- cm["0","0"] / sum(cm[,"0"])
(f1_manual <- 2 * precision * recall / (precision + recall))

# Generate ROC object for AUC score and plot
roc_object_xgb <- roc(testing_set$Diagnosis, xgb_pred, levels = c("1", "0"))
auc(roc_object_xgb)
plot(roc_object_xgb, legacy.axes = TRUE)
coords(roc_object_xgb, "best")
plot(roc_object_xgb,
     col="blue",
     lwd=3,
     print.auc=TRUE,
     legacy.axes=TRUE)
abline(a=0,b=1,lty=2,col="gray")

# ROC object summary for feature importance
xgb_importance <- xgb.importance(bcancer_XGB_model)
xgb.ggplot.importance(xgb_importance,
                      top_n = 35) + 
  theme_minimal()


################# Feature-reduced Random Forest ##################

# Remove infinite values and add Diagnosis as a factor column to data for RF model
bcancer_rrf <- log_bcancer[apply(log_bcancer[2:31], 1, function(x) all(is.finite(x))), ]
drop_cols <- c(7, 10, 11, 17, 20, 21, 27, 30, 31)
bcancer_rrf <- bcancer_rrf[, -drop_cols]
bcancer_rrf$Diagnosis <- as.factor(bcancer_rrf$Diagnosis)
bcancer_rrf$Diagnosis <- as.numeric(bcancer_rrf$Diagnosis) - 1
bcancer_rrf$Diagnosis <- factor(bcancer_rrf$Diagnosis, levels = c(0,1))

# 70/30 split of training and testing data
set.seed(33)
train_index_all <- sample(1:nrow(bcancer_rrf), 0.7 * nrow(bcancer_rrf))
str(train_index_all)
training_bcancer_all <- bcancer_rrf[train_index_all, ]
training_bcancer_all <- training_bcancer_all[, -1]
testing_bcancer_all <- bcancer_rrf[-train_index_all, ]
testing_bcancer_all <- testing_bcancer_all[, -1]

# Train the model
rrf_model <- randomForest(Diagnosis ~ .,
                          data = training_bcancer_all,
                          ntree = 500,
                          importance = T)

# Generate predictions
rrf_predictions <- predict(rrf_model, testing_bcancer_all)
rrf_predictions <- as.numeric(as.character(rrf_predictions))

# View confusion matrix
cm <- confusionMatrix(as.factor(rrf_predictions), testing_bcancer_all$Diagnosis)

# View rankings of importance for accuracy of predictions
importance(rrf_model)
num_var <- ncol(bcancer_rrf) - 1
varImpPlot(rrf_model, n.var = nrow(rrf_model$importance), main = "Variable Importance Plot based on Random Forest")

# Manual F1 score
precision <- cm$table["0","0"] / sum(cm$table["0",])
recall    <- cm$table["0","0"] / sum(cm$table[,"0"])
(f1_manual <- 2 * precision * recall / (precision + recall))

# Generate ROC object for AUC score and plot
roc_object_rrf <- roc(testing_bcancer_all$Diagnosis, rrf_predictions)
auc(roc_object_rrf)
plot(roc_object_rrf,
     col="blue",
     lwd=3,
     print.auc=TRUE,
     legacy.axes=TRUE)
abline(a=0,b=1,lty=2,col="gray")


################################ Feature-Reduced XGBoost ###############################

# Remove infinite values and add Diagnosis as a factor column to data for XGB model, removing 9 columns
bcancer_rxgb <- log_bcancer[apply(log_bcancer[2:31], 1, function(x) all(is.finite(x))), ]
drop_cols <- c(7, 10, 11, 17, 20, 21, 27, 30, 31)
bcancer_rxgb <- bcancer_rxgb[, -drop_cols]
bcancer_rxgb$Diagnosis <- factor(bcancer_rxgb$Diagnosis, levels = c("B", "M"))
bcancer_rxgb$Diagnosis <- as.numeric(bcancer_rxgb$Diagnosis) - 1

# 70/30 split of training and testing data
set.seed(409)
xgb_split <- sample.split(bcancer_rxgb$Diagnosis, SplitRatio = 0.7)
training_set <- subset(bcancer_rxgb, xgb_split == T)
training_set <- training_set[, -1]
testing_set <- subset(bcancer_rxgb, xgb_split == F)
testing_set <- testing_set[, -1]
xgb_train_x <- model.matrix(Diagnosis ~ ., data = training_set)
xgb_test <- model.matrix(Diagnosis ~ ., data = testing_set)
xgb_train_label <- training_set$Diagnosis
final_xgb_train <- xgb.DMatrix(data = xgb_train_x[,-1], label = training_set$Diagnosis)
final_xgb_test <- xgb.DMatrix(data = xgb_test[,-1], label = testing_set$Diagnosis)

# Run XGB model
bcancer_rXGB_model <- xgb.train(data = final_xgb_train, 
                                params = list(objective = "binary:logistic",
                                              eta = 0.25), 
                                nrounds = 200)

# Generate predictions, convert them to binary format
rxgb_pred <- predict(bcancer_rXGB_model, final_xgb_test)
hist(rxgb_pred)
rxgb_pred <- ifelse(rxgb_pred > 0.5, 1, 0)

# View matrix of predictions and actual results
confusionMatrix(as.factor(rxgb_pred), as.factor(testing_set$Diagnosis))
(cm <- table(Predicted = as.factor(rxgb_pred), Actual = as.factor(testing_set$Diagnosis)))

# Manual F1 value
precision <- cm["0","0"] / sum(cm["0",])
recall    <- cm["0","0"] / sum(cm[,"0"])
(f1_manual <- 2 * precision * recall / (precision + recall))

# Generate ROC object for AUC score and plot
roc_object_rxgb <- roc(testing_set$Diagnosis, rxgb_pred, levels = c("1", "0"))
roc_object_rxgb$levels
roc_object_rxgb$direction
auc(roc_object_rxgb)
plot(roc_object_rxgb, legacy.axes = TRUE)
coords(roc_object_rxgb, "best")
plot(roc_object_rxgb,
     col="blue",
     lwd=3,
     print.auc=TRUE,
     legacy.axes=TRUE)
abline(a=0,b=1,lty=2,col="gray")

# ROC object summary for feature importance
rxgb_importance <- xgb.importance(bcancer_rXGB_model)
xgb.ggplot.importance(rxgb_importance) + 
  theme_minimal()

