library(cito)
library(torch)
library(torchaudio)
library(viridis)

set.seed(42)

# Parameters
n_seconds <- 5
sample_rate <- 32000
n_fft <- 400
win_length <- n_fft
hop_length <- win_length %/% 2
n_mels <- 128
n_time_bins <- ((n_seconds*sample_rate - win_length)%/%hop_length)+3
path_to_data <- "data/birdclef2024/"

mel_transformer <- transform_mel_spectrogram(sample_rate = sample_rate,
                                             n_fft = n_fft,
                                             win_length = win_length,
                                             hop_length = hop_length,
                                             n_mels = n_mels)

metadata <- read.csv(paste0(path_to_data, "train_metadata.csv"))

classes <- levels(factor(metadata[,"primary_label"]))

metadata <- metadata[apply(metadata, 1, function(x) !anyNA(x[c("primary_label", "type", "longitude", "latitude", "rating", "filename")])),]

unique_types <- c()
for(i in 1:nrow(metadata)) {
  txt <- tolower(metadata[i,"type"])
  txt <- substring(txt,3,nchar(txt)-2)
  
  for(type in strsplit(txt,"', '")[[1]]) {
    if(nchar(type) == 0) type <- " "
    if(!type %in% unique_types) unique_types <- c(unique_types, type)
  }
}

types_mat <- matrix(0, nrow(metadata), length(unique_types))
colnames(types_mat) <- unique_types
for(i in 1:nrow(metadata)) {
  txt <- tolower(metadata[i,"type"])
  txt <- substring(txt,3,nchar(txt)-2)
  
  for(type in strsplit(txt,"', '")[[1]]) {
    if(nchar(type) == 0) type <- " "
    types_mat[i, type] <- 1
  }
}

indices <- which((types_mat[,"call"]==1 | types_mat[,"song"]==1) & metadata[,"rating"] >= 3)





spectra <- array(0, dim=c(length(indices), 1, n_mels, n_time_bins))
rm_indices <- c()
i <- 0
for(filename in metadata[indices, "filename"]) {
  i <- i+1
  if(i %% 100 == 0) print(i)
  audio <- torchaudio_load(paste0(path_to_data, "train_audio/", filename))
  if(attr(audio, "sample_rate") != sample_rate) {
    print(paste0(i, ": Audio file has wrong sample rate: ", filename))
    rm_indices <- c(rm_indices, i)
    next
  }
  if(dim(audio)[1] != 1) {
    print(paste0(i, ": Audio file has too many channels: ", filename))
    rm_indices <- c(rm_indices, i)
    next
  }
  if(dim(audio)[2] < n_seconds*sample_rate) {
    print(paste0(i, ": Audio file is too short: ", filename))
    rm_indices <- c(rm_indices, i)
    next
  }
  audio_torch <- transform_to_tensor(audio)[[1]][,1:(n_seconds*sample_rate)]
  spectra[i,,,] <- as.array(mel_transformer(audio_torch))
}

spectra <- log(spectra+1)

a <- spectra[1,1,,]
image(t(a[nrow(a):1,]), col = viridis(1000), axes=F)


if(length(rm_indices) != 0) {
  indices <- indices[-rm_indices]
  spectra <- spectra[-rm_indices,,,,drop=F]
}

labels <- factor(metadata[indices, "primary_label"], levels = classes)

dim(spectra)
length(labels)
length(levels(labels))

# min(table(labels))
# hist(table(labels), max(table(labels)))


architecture <- create_architecture(conv(64,7,2,c(3,2)), maxPool(2), conv(128,3,2,1), conv(256,3,2,1), conv(128,1), conv(256,3,2,1), conv(512,3,2,1), conv(256,1), avgPool(c(2,13)), linear(256), 
                                    default_activation = "relu", default_normalization = T, default_dropout = 0.3)
print(architecture, c(1,128,801), 182)

cnn.fit <- cnn(X=spectra,
               Y=labels,
               architecture = architecture,
               loss = "softmax",
               validation = 0.1,
               burnin = Inf,
               epochs = 300,
               early_stopping = 30,
               lr_scheduler = config_lr_scheduler("reduce_on_plateau"),
               device = "cuda")

# cnn.fit <- cnn(X=array(runif(24*24*100), dim = c(100,1,24,24)),
#     Y=factor(sample(c("a","b","d","e"), 100, replace=T), levels = c("a","b","c","d","e")),
#     architecture = create_architecture(conv(),linear()),
#     loss = "softmax", epochs = 3, validation=0.1, device="cpu")

#saveRDS(cnn.fit, file = "analysis/results/CNN1.rds")
#cnn.fit <- readRDS(file = "analysis/results/CNN1.rds")

pred_classes <- predict(cnn.fit, newdata = cnn.fit$data$X[cnn.fit$data$validation,,,,drop=F], type = "class")
pred_response <- predict(cnn.fit, newdata = cnn.fit$data$X[cnn.fit$data$validation,,,,drop=F], type = "response")
true <- cnn.fit$data$Y[cnn.fit$data$validation]

library(MLmetrics)


Accuracy(pred_classes, true)

library(pROC)

tab <- table(cnn.fit$data$Y)

color_palette <- colorRampPalette(c("white", "blue"))
colors <- color_palette(max(tab))

{
  layout(matrix(1:2,nrow=1),widths=c(0.9,0.1))
  par(mar=c(5.1,4.1,4.1,2.1))
  add <- F
  for(class in names(sort(tab, decreasing = F))) {
    if(!any(true == class)) next
    curve <- roc(as.numeric(true == class), pred_response[, class])
    plot(curve, col=colors[tab[class]], add=add, main="ROC curves")
    add <- T
  }
  
  par(mar=c(5.1,0.5,4.1,0.5))
  image(1, 1:max(tab), matrix(1:max(tab), nrow=1), col=colors, axes=F, xlab="", ylab="")
  # Add axis on the right for the legend
  axis(2, at=round(c(1, max(tab)/4, max(tab)/2, 3*max(tab)/4, max(tab))))
  box()
}

