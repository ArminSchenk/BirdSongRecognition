library(cito)
library(torch)

# Set GPU
Sys.setenv(CUDA_VISIBLE_DEVICES=3)

# Seed
set.seed(42)
torch::torch_manual_seed(42)

# Load data
if(!exists("spectra")) spectra <- readRDS("data/spectra.rds")
if(!exists("metadata")) metadata <- readRDS("data/metadata.rds")
if(!exists("folds")) folds <- readRDS("data/folds.rds")

# Multimodal Neural Networks
pred_train <- vector("list", 5)
pred_valid <- vector("list", 5)
for (i in 1:5) {
  mmn.fit <- mmn(metadata[,"primary_label"] ~ cnn(X=spectra, architecture=create_architecture(transfer("mobilenet_v2", freeze=F))) 
                 + dnn(~ longitude + latitude, data=metadata),
                 dataList = list(spectra=spectra[unlist(folds[-i]),,,,drop=F],
                                 metadata=metadata[unlist(folds[-i]),]),
                 loss = "softmax",
                 validation = 0.1,
                 epochs = 100,
                 early_stopping = 3,
                 device = "cuda",
                 batchsize = 16)
  
  mmn.fit$data <- list(ylvls=levels(metadata[,"primary_label"]))
  
  gc()
  torch::cuda_empty_cache()
  gc()
  torch::cuda_empty_cache()
  
  pred_train[[i]] <- predict(mmn.fit, newdata=list(spectra=spectra[unlist(folds[-i]),,,,drop=F], metadata=metadata[unlist(folds[-i]),]), type = "response")
  pred_valid[[i]] <- predict(mmn.fit, newdata=list(spectra=spectra[folds[[i]],,,,drop=F], metadata=metadata[folds[[i]],]), type = "response")

  saveRDS(mmn.fit, file = paste0("analysis/results/models/MMN_fold", i, ".rds"))
  
  rm(mmn.fit)
  gc()
  torch::cuda_empty_cache()
  gc()
  torch::cuda_empty_cache()
}

saveRDS(pred_train, file = "analysis/results/predictions/MMN_pred_train.rds")
saveRDS(pred_valid, file = "analysis/results/predictions/MMN_pred_valid.rds")