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

indices <- which(types_mat[,"call"]==1 & metadata[,"rating"] >= 5)





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


architecture <- create_architecture(conv(4,5), maxPool(2), conv(8,3), maxPool(2), conv(16,3), maxPool(4), linear(100), linear(100),
                                    default_activation = "selu", default_normalization = T, default_dropout = 0.3)
architecture <- create_architecture(conv(),maxPool(20),linear(10))
print(architecture, c(1,128,801), 182)

cnn.fit <- cnn(X=spectra,
               Y=labels,
               architecture = architecture,
               loss = "softmax",
               validation = 0.1,
               burnin = 100,
               epochs = 1,
               device = "cpu")


# cnn(X=array(runif(24*24*100), dim = c(100,1,24,24)),
#     Y=factor(sample(c("a","b","c"), 100, replace=T), levels = c("a","b","c","d")),
#     architecture = create_architecture(conv(),linear()),
#     loss = "softmax", epochs = 3, validation=0.1, device="cuda")

