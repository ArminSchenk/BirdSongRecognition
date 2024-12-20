library(ggplot2)
library(ggbeeswarm)
library(pROC)

get_auc <- function(pred, metadata) {
  pred <- do.call(rbind, pred)
  true <- metadata[rownames(pred), "primary_label"]
  auc <- numeric(length(levels(metadata[,"primary_label"])))
  names(auc) <- levels(metadata[,"primary_label"])
  for(class in names(auc)) {
    auc[class] <- auc(roc(as.numeric(true==class), pred[,class]))
  }
  return(auc)
}

# Load data
if(!exists("metadata")) metadata <- readRDS("data/metadata.rds")
# DNN_pred_train <- readRDS("analysis/results/predictions/DNN_pred_train.rds")
# CNN_pred_train <- readRDS("analysis/results/predictions/CNN_pred_train.rds")
# MMN_pred_train <- readRDS("analysis/results/predictions/MMN_pred_train.rds")
#DNN_pred_valid <- readRDS("analysis/results/predictions/DNN_pred_valid.rds")
CNN_pred_valid <- readRDS("analysis/results/predictions/CNN_pred_valid.rds")
MMN_pred_valid <- readRDS("analysis/results/predictions/MMN_pred_valid.rds")

tab <- sort(table(metadata[["primary_label"]]))
classes <- names(tab)

# DNN_auc_train <- get_auc(DNN_pred_train, metadata)[classes]
# CNN_auc_train <- get_auc(CNN_pred_train, metadata)[classes]
# MMN_auc_train <- get_auc(MMN_pred_train, metadata)[classes]
#DNN_auc_valid <- get_auc(DNN_pred_valid, metadata)[classes]
CNN_auc_valid <- get_auc(CNN_pred_valid, metadata)[classes]
MMN_auc_valid <- get_auc(MMN_pred_valid, metadata)[classes]

# Example Data: Replace these with your actual AUC values
a <- min(c(CNN_auc_valid, MMN_auc_valid))
b <- max(c(CNN_auc_valid, MMN_auc_valid))
DNN_auc_valid <- runif(177, a, b)
DNN_auc_train <- runif(177, a, b)
CNN_auc_train <- runif(177, a, b)
MMN_auc_train <- runif(177, a, b)

# Combine data into a data frame
auc_data <- data.frame(
  Model = factor(rep(c("DNN", "CNN", "MMN"), each = 354), levels = c("DNN", "CNN", "MMN")),
  DataType = rep(c(rep("Training", 177), rep("Validation", 177)), 3),
  AUC = c(DNN_auc_train, DNN_auc_valid, CNN_auc_train, CNN_auc_valid, MMN_auc_train, MMN_auc_valid),
  ClassSize = rep(as.numeric(tab), 6)
)

pdf("analysis/results/figures/auc_beehive.pdf", width = 20, height = 10)

beehive_plot <- ggplot(auc_data, aes(x = DataType, y = AUC, color = ClassSize)) +
  geom_beeswarm(cex = 2, alpha = 1) + 
  facet_wrap(~Model, scales = "fixed") +
  scale_color_gradient(low = "#FFDDDD", high = "red") +
  theme_minimal() +
  labs(
    #title = "Beehive Plot of AUC Values by Class Size",
    x = "",
    y = "AUC",
    color = "Class Size"
  ) +
  theme(
    text = element_text(size = 14),
    strip.text = element_text(size = 12, face = "bold"),
    panel.border = element_rect(color = "black", fill = NA, size = 1) # Add a black border
    axis.title.y = element_text(margin = margin(r = 100), size = 14) # Move y-axis label left
  )

# Print the plot
print(beehive_plot)

dev.off()

