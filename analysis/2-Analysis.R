library(cito)

# Seed
set.seed(42)

# Load data
if(!exists("spectra")) readRDS("data/spectra.rds")
if(!exists("metadata")) readRDS("data/metadata.rds")

stratified_kfold <- function(data, target_col, k = 5) {
  # Split data by the target column
  split_data <- split(data, data[, target_col])
  
  # Initialize folds
  folds <- vector("list", k)
  
  # Assign samples to each fold
  for (class_data in split_data) {
    fold_indices <- cut(sample(1:nrow(class_data)), breaks = k, labels = FALSE)
    
    for (i in 1:k) {
      if (is.null(folds[[i]])) {
        folds[[i]] <- rownames(class_data)[fold_indices == i]
      } else {
        folds[[i]] <- c(folds[[i]], rownames(class_data)[fold_indices == i])
      }
    }
  }
  
  return(folds)
}

folds <- stratified_kfold(metadata, "primary_label")

cnn.fit <- cnn(X=spectra[folds[[1]],,,,drop=F],
               Y=metadata[folds[[1]], "primary_label"],
               architecture = create_architecture(transfer("mobilenet_v2", freeze=F)),
               loss = "softmax",
               validation = 0.1,
               epochs = 1,
               device = "cuda")

models <- c("alexnet", "inception_v3", "mobilenet_v2", "resnet101", "resnet152",
  "resnet18", "resnet34", "resnet50", "resnext101_32x8d", "resnext50_32x4d", "vgg11",
  "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
  "wide_resnet101_2", "wide_resnet50_2")

for(model in models) {
  eval(parse(text = paste0("torch_model <- torchvision::model_", model, "(pretrained = FALSE)")))
  tmp <- torch_model$modules
  while(TRUE) {
    tmp <- tmp[[1]]
    if(inherits(tmp, "nn_conv2d")) {
      print(paste0(model, ": ", tmp$out_channels))
      print(paste0(model, ": ", paste0(tmp$kernel_size, collapse = "x")))
      print(paste0(model, ": ", paste0(tmp$stride, collapse = "x")))
      print(paste0(model, ": ", paste0(tmp$padding, collapse = "x")))
      print(paste0(model, ": ", paste0(tmp$dilation, collapse = "x")))
      bias <- ifelse(is.null(tmp$bias), TRUE, FALSE)
      print(paste0(model, ": ", bias))
      print(paste0(model, ": ", tmp$groups))
      print(paste0(model, ": ", tmp$padding_mode))
      
      break
    }
  }
}



