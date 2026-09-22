# PSPS-Voice
Codes for processing audio WAV files to create feature vectors and spectrograms, and estimate AUC in random iterations with 2/3 training and 1/3 validation partitions.

We have been exploring the use of voice features and spectrograms collected from people with Persistent Spinal Pain Syndrome (PSPS) and controls to correctly classify between PSPS and controls. This work is described in the following manuscript:

Rahmatallah Y, Kemp AS, Iyer A, Wilkerson C, Shegena EA, Petersen E, Tobey-Moore L. Machine learning approaches for the identification of pain from telephone voice samples of patients diagnosed with Persistent Spinal Pain Syndrome. Submitted to Scientific Reports (October, 2026) at currently under revision. 

Feature vectors and spectrogram images were produced from voice recordings of study participants who enunciate the sustained vowels /a/ (ah), /i/ (ee), and /u/ (oo). Participant voice recordings are available from figshare as “Voice Samples for Patients with Persistent Spinal Pain Syndrome and controls”, #paste DOI here#. Recordings from participants were collected under the University of Arkansas for Medical Sciences (UAMS) Institutional IRB #297377. 

This repository provides the following R codes:
1.	Calculate_SpectralFeatures.R: Calculate mean and standard deviation summary feature vectors for 4 types of spectral features: Linear prediction coding coefficients (LPC), log-area ratio coefficients (LAR), Cepstral coefficients (Cep), and mel-frequency cepstral coefficients (MFCC).
2.	Create_Spectrograms.R: Generate mel-scale spectrograms and save them in jpg file format.
3.	estimate_ML_AUC.R: Estimate area under the receiver operational curve (AUC) in three machine learning classifiers: Random Forest (RF), Ridge Logistic Regression (LR), and Support Vector Machine (SVM). Samples are randomly split into 2/3 training and 1/3 validation partitions.

Spectrogram images are analyzed using the # paste name here# Jupyter notebook. An Inception V3 CNN pre-trained on Imagenet is adapted to this problem of extracting features from images and classifying the speaker as either a control or a PSPS patient. The original classification stage of the Inception model was replaced with four custom layers: batch normalization, 2 dense layers (1024 nodes, relu activation)and a final dense layer (2 classes, softmax activation) to create a multi-layer perceptron classifier stage. This classifier was trained using the data cited above to implement transfer learning. It is important to organize the spectra images into the directory structure required by the Keras ImageDataGenerator class when using the class_mode- 'categorical' option, i.e., the data_path points to a directory with 2 sub directories, one with Healthy Control spectra and one with PwPD spectra. (see https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720). The associated Anaconda environment is provided for reference, environment.yaml.
 
A Jupyter notebook containing the code used to extract acoustic features using Parselmouth a package that runs Praat in Python. Praat can be found here: PraatScripts on GitHub. Our notebook measures pitch, standard deviation of pitch, harmonics-to-noise ratio (HNR), jitter, shimmer, and formants from the original .wav files. 
(https://github.com/drfeinberg/PraatScripts)
