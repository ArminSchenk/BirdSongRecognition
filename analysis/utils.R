# Function to get stratified k-fold cross-validation indices
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

# One vs All AUCs for all classes
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