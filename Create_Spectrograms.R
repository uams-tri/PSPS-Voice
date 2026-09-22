###################################################################
# R code to generate spectrogram images from the UAMS audio files #
###################################################################

# Copyright (C) 2026 University of Arkansas for Medical Sciences
# Author: Yasir Rahmatallah, yrahmatallah@uams.edu
# Licensed under the Apache License, Version 2.0
# You may not use this file except in compliance with the License
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# Code was tested using R version 4.4.2, and the following package versions:
# av_0.9.4, tuneR_1.4.7, oce_1.8-3, signal_1.8-1, viridis_0.6.5                    

library(av)
library(tuneR)
library(oce)
library(signal)
library(viridis)

# function to trim silence parts in the WAV files
trim_ends <- function(x, w=100, thr=1, start=1){
len <- length(x)
nseg <- floor(len/w)
ste <- numeric(nseg)
for(q in 1:nseg) ste[q] <- sum(x[((q-1)*w+1):(q*w)]^2)
env <- rep(ste, each=w)
check <- (env>thr)+0
rr <- rle(check)
if(length(rr$lengths) > 1) # apply when parts of the recording pass the threshold
	{
	indmax <- which.max(rr$lengths * rr$values)
	if(indmax>1)
		{
		st <- sum(rr$lengths[1:(indmax-1)]) + start
		en <- sum(rr$lengths[1:indmax])
		}
	if(indmax==1)
		{
		st <- start
		en <- rr$lengths[1]
		}
	} else {st <- 1; en <- len} # apply when all the recording passes the threshold
return(c(st, en))
}


time.limit <- 1 # minimum voiced segment length in seconds
file_audio_path <- "path to where the wav files were saved"

# find the names of wav files in the selected folder
fL <- list.files(path=file_audio_path, pattern=".wav$")
# remove '.wav' from the end of file names
fL <- gsub(fL, pattern=".wav", replacement="")

# start a loop
for(file.name in fL){
	file <- paste(file_audio_path, "/", file.name, ".wav", sep="")
	train_audio <- readWave(file)
	audio.nor <- tuneR::normalize(train_audio, unit="32", pcm=FALSE)
	x <- audio.nor@left
	fs <- train_audio@samp.rate
	duration <- length(x) / fs
	
	ends <- trim_ends(x, w=100, thr=1, start=1000)
	if((ends[2] - ends[1]) < (fs*time.limit))
		{
		print(paste("File ", file.name, " is < ", time.limit," sec, skipped!", sep=""), quote=F)
		next
		} else x <- x[ends[1]:(ends[1]+fs*time.limit-1)]
	audio.nor@left <- x
	audio.nor@samp.rate <- fs

	# create mel-scale spectrogram
	sample_mp3 <- transform_to_tensor(audio.nor)
	w <- 512
	mel_specgram2 <- transform_mel_spectrogram(sample_rate=sample_mp3[[2]],
	n_fft = 1024, win_length = w, hop_length = (w*0.1),
	f_min = 0, f_max = 4000, pad = 0, n_mels = 256,
	window_fn = torch::torch_hann_window,
	power = 2, normalized = FALSE)(sample_mp3[[1]])
	specgram_as_array2 <- as.array(mel_specgram2$log2()[1]$t())

	# create one folder to save spectrogram images
	dir.create(path=paste("path to where you save spectrogram image", 
	"/linearSpec", sep=""))
	fileJPG <- paste("path to the created folder earlier", "/", file.name, ".jpg", sep="")
	
	# plot one spectrogram image in a jpeg file. Image is 2-by-2 inches with 300 dpi resolution
	jpeg(filename=fileJPG, res=300, width=2, height=2, units="in", pointsize=1, quality=100)

	par(mfrow=c(1,1), mar=c(0,0,0,0))
	image(specgram_as_array2[,1:ncol(specgram_as_array2)], 
	col=oce.colorsViridis(256), xaxt="n", yaxt="n")
	dev.off()

	} # end of loop
