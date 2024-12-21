library(ggplot2)
library(ggbeeswarm)
library(pROC)

# Seed
set.seed(42)

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
DNN_pred_train <- readRDS("analysis/results/predictions/DNN_pred_train.rds")
CNN_pred_train <- readRDS("analysis/results/predictions/CNN_pred_train.rds")
# MMN_pred_train <- readRDS("analysis/results/predictions/MMN_pred_train.rds")
DNN_pred_valid <- readRDS("analysis/results/predictions/DNN_pred_valid.rds")
CNN_pred_valid <- readRDS("analysis/results/predictions/CNN_pred_valid.rds")
# MMN_pred_valid <- readRDS("analysis/results/predictions/MMN_pred_valid.rds")

tab <- sort(table(metadata[["primary_label"]]))
classes <- names(tab)

DNN_auc_train <- get_auc(DNN_pred_train, metadata)[classes]
CNN_auc_train <- get_auc(CNN_pred_train, metadata)[classes]
# MMN_auc_train <- get_auc(MMN_pred_train, metadata)[classes]
DNN_auc_valid <- get_auc(DNN_pred_valid, metadata)[classes]
CNN_auc_valid <- get_auc(CNN_pred_valid, metadata)[classes]
# MMN_auc_valid <- get_auc(MMN_pred_valid, metadata)[classes]

# Example Data: Replace these with your actual AUC values
a <- min(c(CNN_auc_valid, DNN_auc_valid, CNN_auc_train, DNN_auc_train))
b <- max(c(CNN_auc_valid, DNN_auc_valid, CNN_auc_train, DNN_auc_train))
MMN_auc_train <- runif(177, a, b)
MMN_auc_valid <- runif(177, a, b)


# Combine data into a data frame
auc_data <- data.frame(
  Model = factor(rep(c("DNN", "CNN", "MMN"), each = 354), levels = c("DNN", "CNN", "MMN")),
  DataType = rep(c(rep("Training", 177), rep("Validation", 177)), 3),
  AUC = c(DNN_auc_train, DNN_auc_valid, CNN_auc_train, CNN_auc_valid, MMN_auc_train, MMN_auc_valid),
  ClassSize = rep(as.numeric(tab), 6)
)

pdf("analysis/results/figures/auc_beehive.pdf", width = 20, height = 10)

beehive_plot <- ggplot(auc_data, aes(x = DataType, y = AUC, color = ClassSize)) +
  geom_jitter(width = 0.3, height = 0, size = 2, alpha = 0.8) +
  geom_boxplot(
    aes(group = interaction(DataType, Model)), 
    width = 0.7, 
    outlier.shape = NA, # Avoid overlapping outliers with jitter
    alpha = 0 # Make boxplot semi-transparent
  ) +
  facet_wrap(~Model, scales = "fixed", nrow = 1, strip.position = "top") +
  scale_color_viridis_c(option = "viridis", direction = -1) +
  theme_minimal() +
  labs(
    x = NULL, # Remove x-axis label
    y = "AUC", # Y-axis label
    color = "Class Size"
  ) +
  theme(
    text = element_text(size = 14),
    strip.text = element_text(size = 16, face = "bold"), # Increase model type label size
    axis.title.y = element_text(size = 20, margin = margin(r = 10)), # Increase y-axis label size
    axis.text.x = element_text(size = 14), # Increase x-axis label size
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    panel.spacing = unit(1, "lines")
  )

# Print the plot
print(beehive_plot)

dev.off()

