library(torchaudio)
library(torch)

# Seed
set.seed(42)

# Mel spectrogram parameters
n_seconds <- 10
sample_rate <- 32000
n_fft <- 1024
win_length <- n_fft
hop_length <- win_length %/% 2
n_mels <- 128
f_min <- 40
f_max <- 15000
power <- 2

# Data path
path_to_data <- "data/birdclef2024/"

# Thresholds to filter data
rating_treshold <- 3
classes_treshold <- 5

target_length <- n_seconds*sample_rate
n_time_bins <- ((target_length - win_length)%/%hop_length)+3

# Load metadata
metadata <- read.csv(paste0(path_to_data, "train_metadata.csv"))
metadata <- metadata[,c("primary_label", "longitude", "latitude", "rating", "filename")]

# Filter samples with missing data
metadata <- metadata[!apply(metadata, 1, anyNA),]

# Filter audio files with bad rating
metadata <- metadata[which(metadata[,"rating"] >= rating_treshold),]

# Load audio and generate mel-spectrograms
mel_transformer <- transform_mel_spectrogram(sample_rate = sample_rate,
                                             n_fft = n_fft,
                                             win_length = win_length,
                                             hop_length = hop_length,
                                             f_min = f_min,
                                             f_max = f_max,
                                             n_mels = n_mels,
                                             power = power)

spectra <- array(0, dim=c(nrow(metadata), 1, n_mels, n_time_bins), dimnames = list(rownames(metadata)))
rm_indices <- c()
i <- 0
print(nrow(metadata))
for(filename in metadata[,"filename"]) {
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
  audio_torch <- transform_to_tensor(audio)[[1]]
  audio_length <- dim(audio_torch)[2]
  dif_length <- abs(target_length - audio_length)
  
  # Padding or Cropping
  if(audio_length < target_length) {
    left_pad <- sample.int(dif_length, 1)
    right_pad <- dif_length - left_pad
    audio_torch <- torch_cat(list(torch_zeros(1, left_pad), audio_torch, torch_zeros(1, right_pad)), dim = 2L)
  } else if(audio_length > target_length) {
    idx <- sample.int(dif_length+1, 1)
    audio_torch <- audio_torch[,idx:(idx-1+target_length)]
  }
  
  spectra[i,,,] <- as.array(mel_transformer(audio_torch))
}

# Filter unsuitable audio files
if(length(rm_indices) != 0) {
  metadata <- metadata[-rm_indices,]
  spectra <- spectra[-rm_indices,,,,drop=F]
}

# Filter exceptionally rare classes
classes <- names(which(table(metadata[,"primary_label"]) >= classes_treshold))
indices <- which(metadata[,"primary_label"] %in% classes)
metadata <- metadata[indices,]
spectra <- spectra[indices,,,,drop=F]

# Logarithm to mimic human auditory perception
spectra <- log10(spectra+1)

# Normalize
for(i in 1:nrow(metadata)) {
  max <- max(spectra[i,,,])
  min <- min(spectra[i,,,])
  if(max-min >= 1e-10) spectra[i,,,] <- (spectra[i,,,] - min) / (max - min)
  else spectra[i,,,] <- spectra[i,,,] - min
}

metadata[,"primary_label"] <- factor(metadata[,"primary_label"])

saveRDS(spectra, file = "data/spectra.rds")
saveRDS(metadata, file = "data/metadata.rds")

rm(audio,
   audio_length,
   audio_torch,
   classes,
   classes_treshold,
   dif_length,
   f_max,
   f_min,
   filename,
   hop_length,
   i,
   idx,
   indices,
   left_pad,
   max,
   mel_transformer,
   min,
   n_fft,
   n_mels,
   n_seconds,
   n_time_bins,
   path_to_data,
   power,
   rating_treshold,
   right_pad,
   rm_indices,
   sample_rate,
   target_length,
   win_length)

