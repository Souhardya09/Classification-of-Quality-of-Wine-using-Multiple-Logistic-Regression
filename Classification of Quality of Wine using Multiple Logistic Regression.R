library(ggplot2)
library(dplyr)
library(reshape2)
library(caTools)
library(caret)
library(pROC)

# -------------------
# 1. Load & prepare data
# -------------------
wine <- read.csv("C:/Users/SOUHARDYA/Documents/FIFA 19/wine.csv")

nrow(wine)
# Binary target: quality >= 6 → 1 else 0
wine$quality_binary <- ifelse(wine$quality >= 6, 1, 0)

# -------------------
# 2. EDA with ggplot2
# -------------------

## Distribution of wine quality
ggplot(wine, aes(x = factor(quality))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Wine Quality", x = "Quality", y = "Count") +
  theme_minimal()

## Binary distribution
ggplot(wine, aes(x = factor(quality_binary), fill = factor(quality_binary))) +
  geom_bar() +
  labs(title = "Low vs High Quality Wines", x = "Quality Group", y = "Count") +
  scale_fill_manual(values = c("tomato", "seagreen")) +
  theme_minimal()



## Correlation heatmap (drop binary column)
num_data <- wine %>% select(where(is.numeric), -quality_binary)
corr <- cor(num_data, use = "pairwise.complete.obs")
melted_corr <- melt(corr, na.rm = TRUE)

ggplot(melted_corr, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "red", high = "green", mid = "white",
                       midpoint = 0, limit = c(-1,1)) +
  labs(title = "Correlation Heatmap of Features") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

## Correlation of features with binary target
target_corr <- cor(wine %>% select(where(is.numeric)), use = "pairwise.complete.obs")
quality_corr <- sort(target_corr[,"quality_binary"], decreasing = TRUE)

quality_corr_df <- data.frame(
  Feature = names(quality_corr),
  Correlation = quality_corr
)

ggplot(quality_corr_df %>% filter(Feature != "quality_binary"),
       aes(x = reorder(Feature, Correlation), y = Correlation, fill = Correlation)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  labs(title = "Correlation of Features with Wine Quality (Binary)",
       x = "Feature", y = "Correlation") +
  theme_minimal()

## Boxplot: Alcohol vs Quality
ggplot(wine, aes(x = factor(quality), y = alcohol, fill = factor(quality))) +
  geom_boxplot() +
  labs(title = "Alcohol Content by Wine Quality", x = "Quality", y = "Alcohol") +
  theme_minimal()

## Boxplot: Volatile acidity vs Quality
ggplot(wine, aes(x = factor(quality), y = volatile.acidity, fill = factor(quality))) +
  geom_boxplot() +
  labs(title = "Volatile Acidity by Wine Quality", x = "Quality", y = "Volatile Acidity") +
  theme_minimal()

## Density: Alcohol by Binary Quality
ggplot(wine, aes(x = alcohol, fill = factor(quality_binary))) +
  geom_density(alpha = 0.6) +
  labs(title = "Distribution of Alcohol by Quality Group", x = "Alcohol", y = "Density") +
  theme_minimal()

## Scatter: Alcohol vs Density
ggplot(wine, aes(x = density, y = alcohol, color = factor(quality_binary))) +
  geom_point(alpha = 0.6) +
  labs(title = "Alcohol vs Density by Quality Group", x = "Density", y = "Alcohol") +
  theme_minimal()




ggplot(wine, aes(x = factor(quality_binary), fill = factor(quality_binary))) +
  geom_bar() +
  labs(title = "Low vs High Quality Wines", x = "Quality Group", y = "Count") +
  scale_fill_manual(values = c("tomato", "seagreen")) +
  theme_minimal()


# -------------------
# 3. Train/Test Split
# -------------------
set.seed(123)
split <- sample.split(wine$quality_binary, SplitRatio = 0.7)
train <- subset(wine, split == TRUE)
test  <- subset(wine, split == FALSE)

# -------------------
# 4. Logistic Regression
# -------------------
model <- glm(quality_binary ~ . - quality, data = train, family = binomial)

summary(model)
library(car)
vif(model)

model2=glm(quality_binary ~ . - quality-fixed.acidity-residual.sugar-density, data = train, family = binomial)
summary(model2)
vif(model2)
# -------------------
# 5. Evaluation
# -------------------

## Predict probabilities & classes
test$pred_prob <- predict(model2, newdata = test, type = "response")
test$pred_class <- ifelse(test$pred_prob >= 0.616, 1, 0)

## Confusion Matrix
confusionMatrix(as.factor(test$pred_class), as.factor(test$quality_binary))

## ROC Curve
roc_obj <- roc(test$quality_binary, test$pred_prob)

ggroc(roc_obj, color = "blue", size = 1.2) +
  ggtitle(paste("ROC Curve (AUC =", round(auc(roc_obj), 3), ")")) +
  theme_minimal()

## Histogram of predicted probabilities by class
ggplot(test, aes(x = pred_prob, fill = factor(quality_binary))) +
  geom_histogram(position = "identity", bins = 30, alpha = 0.6) +
  labs(title = "Predicted Probability Distribution",
       x = "Predicted Probability (quality >= 6)",
       fill = "True Class") +
  theme_minimal()