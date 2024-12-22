library(ggplot2)
library(ggbeeswarm)
library(pROC)
source("analysis/utils.R")

# Seed
set.seed(42)

# Load data
if(!exists("metadata")) metadata <- readRDS("data/metadata.rds")
DNN_pred_train <- readRDS("analysis/results/predictions/DNN_pred_train.rds")
CNN_pred_train <- readRDS("analysis/results/predictions/CNN_pred_train.rds")
MMN_pred_train <- readRDS("analysis/results/predictions/MMN_pred_train.rds")
DNN_pred_valid <- readRDS("analysis/results/predictions/DNN_pred_valid.rds")
CNN_pred_valid <- readRDS("analysis/results/predictions/CNN_pred_valid.rds")
MMN_pred_valid <- readRDS("analysis/results/predictions/MMN_pred_valid.rds")

tab <- sort(table(metadata[["primary_label"]]))
classes <- names(tab)

# Calculate one-vs-all AUCs
DNN_auc_train <- get_auc(DNN_pred_train, metadata)[classes]
CNN_auc_train <- get_auc(CNN_pred_train, metadata)[classes]
MMN_auc_train <- get_auc(MMN_pred_train, metadata)[classes]
DNN_auc_valid <- get_auc(DNN_pred_valid, metadata)[classes]
CNN_auc_valid <- get_auc(CNN_pred_valid, metadata)[classes]
MMN_auc_valid <- get_auc(MMN_pred_valid, metadata)[classes]

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
    outlier.shape = NA, 
    alpha = 0 
  ) +
  facet_wrap(~Model, scales = "fixed", nrow = 1, strip.position = "top") +
  scale_color_viridis_c(option = "viridis", direction = -1) +
  theme_minimal() +
  labs(
    x = NULL, 
    y = "AUC", 
    color = "Class Size"
  ) +
  theme(
    text = element_text(size = 18),
    strip.text = element_text(size = 24, face = "bold"), 
    axis.title.y = element_text(size = 30, margin = margin(r = 15)), 
    axis.text.x = element_text(size = 20), 
    legend.title = element_text(size = 20), 
    legend.text = element_text(size = 18), 
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    panel.spacing = unit(1, "lines")
  ) +
  guides(
    color = guide_colorbar(barwidth = 2, barheight = 10) 
  )

print(beehive_plot)

dev.off()

# Filter the data for validation set only
validation_data <- subset(auc_data, DataType == "Validation")

pdf("analysis/results/figures/auc_classSize.pdf", width = 20, height = 10)

# Create the plot
plot <- ggplot(validation_data, aes(x = ClassSize, y = AUC, color = Model)) +
  geom_point(alpha = 0.7, size = 2) + 
  geom_smooth(method = "lm", se = FALSE, linetype = "dashed") + 
  scale_x_log10() +
  labs(
    x = "Class Size (Log Scale)",
    y = "AUC",
    color = "Model"
  ) +
  theme_minimal() + 
  theme(
    text = element_text(size = 18),
    strip.text = element_text(size = 24, face = "bold"), 
    axis.title.y = element_text(size = 30, margin = margin(r = 15)), 
    axis.text.x = element_text(size = 20), 
    legend.title = element_text(size = 20), 
    legend.text = element_text(size = 18), 
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    panel.spacing = unit(1, "lines")
  ) 

# Display the plot
print(plot)

dev.off()

