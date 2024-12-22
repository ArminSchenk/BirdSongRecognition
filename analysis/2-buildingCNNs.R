library(cito)
library(torch)

# Set GPU
Sys.setenv(CUDA_VISIBLE_DEVICES=2)

# Seed
set.seed(42)
torch::torch_manual_seed(42)

# Load data
if(!exists("spectra")) spectra <- readRDS("data/spectra.rds")
if(!exists("metadata")) metadata <- readRDS("data/metadata.rds")
if(!exists("folds")) folds <- readRDS("data/folds.rds")

# Convolutional Neural Network
pred_train <- vector("list", 5)
pred_valid <- vector("list", 5)
for (i in 1:5) {
  cnn.fit <- cnn(X=spectra[unlist(folds[-i]),,,,drop=F],
                 Y=metadata[unlist(folds[-i]), "primary_label"],
                 architecture = create_architecture(transfer("mobilenet_v2", freeze=F)),
                 loss = "softmax",
                 validation = 0.1,
                 epochs = 100,
                 early_stopping = 3,
                 device = "cuda",
                 batchsize = 16,
                 plot = FALSE)
  
  cnn.fit$data <- list(ylvls=levels(metadata[,"primary_label"]))
  
  gc()
  torch::cuda_empty_cache()
  
  pred_train[[i]] <- predict(cnn.fit, spectra[unlist(folds[-i]),,,,drop=F], type = "response")
  pred_valid[[i]] <- predict(cnn.fit, spectra[folds[[i]],,,,drop=F], type = "response")

  saveRDS(cnn.fit, file = paste0("analysis/results/models/CNN_fold", i, ".rds"))
  
  rm(cnn.fit)
  gc()
  torch::cuda_empty_cache()
}

saveRDS(pred_train, file = "analysis/results/predictions/CNN_pred_train.rds")
saveRDS(pred_valid, file = "analysis/results/predictions/CNN_pred_valid.rds")
