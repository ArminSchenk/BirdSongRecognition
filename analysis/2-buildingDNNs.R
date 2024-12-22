library(cito)
library(torch)

# Set GPU
Sys.setenv(CUDA_VISIBLE_DEVICES=1)

# Seed
set.seed(42)
torch::torch_manual_seed(42)

# Load data
if(!exists("metadata")) metadata <- readRDS("data/metadata.rds")
if(!exists("folds")) folds <- readRDS("data/folds.rds")

# Dense Neural Networks
pred_train <- vector("list", 5)
pred_valid <- vector("list", 5)
for (i in 1:5) {
  dnn.fit <- dnn(primary_label ~ longitude + latitude,
                 data = metadata[unlist(folds[-i]),],
                 loss = "softmax",
                 validation = 0.1,
                 epochs = 100,
                 early_stopping = 3,
                 device = "cuda",
                 batchsize = 16,
                 plot = FALSE)
  
  dnn.fit$data <- list(ylvls=levels(metadata[,"primary_label"]))
  
  gc()
  torch::cuda_empty_cache()
  
  pred_train[[i]] <- predict(dnn.fit, newdata=metadata[unlist(folds[-i]),], type = "response")
  pred_valid[[i]] <- predict(dnn.fit, newdata=metadata[folds[[i]],], type = "response")

  saveRDS(dnn.fit, file = paste0("analysis/results/models/DNN_fold", i, ".rds"))
  
  rm(dnn.fit)
  gc()
  torch::cuda_empty_cache()
}

saveRDS(pred_train, file = "analysis/results/predictions/DNN_pred_train.rds")
saveRDS(pred_valid, file = "analysis/results/predictions/DNN_pred_valid.rds")
