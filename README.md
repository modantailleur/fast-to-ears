# FAST-TO-EARS : FAST Third-Octave input for Environmental Audio source Recognition System

FAST-TO-EARS enables sound source predictions using fast third-octave recorded datasets. Fast third-octaves should consist of 29 third-octave bins (ranging from 20Hz to 12.5kHz), recorded every 125ms.

Utilizing the transcoding method introduced in [1], it transcodes the third-octave spectro-temporal representations into fine-grained Mel spectrograms. 
These spectrograms then serve as input for PANNs [2], which are large-scale pre-trained classifiers capable of predicting the presence of over 527 different sound sources. Consult the Audioset ontology [3] to identify the sound sources that best suit your project.

## SETUP

First, install the required dependencies using the following command in a new Python 3.9.15 environment:

```
pip install -r requirements.txt
```

Download the pretrained model (PANN ResNet38) by executing the following command:

```
python3 download/download_pretrained_models.py
```

## PANN PREDICTION FROM FAST THIRD-OCTAVES DATASET

Place your h5 file or npy file in the "spectral_data" folder. 
If it is an h5 file, it should contain a dataset named "fast_125ms" at its root.

The npy array from the npy file, or the array from the h5 dataset, should be of size `(nb_of_chunks, nb_of_frames_per_chunks, 29)`. 29 corresponds to the 29 third-octaves between 20Hz and 12.5kHz. As fast third-octaves use windows of 125ms, there are 8 third-octave frames in 1s. Thus, nb_of_frames_per_chunks can be any number above 8, as the transcoder is only able to deal with at least 1-s files. We recommand using a nb_of_frames_per_chunks of 80 or more, as the ResNet38 PANN model used is more performant with 10s audio chunks. If chunks are greater than 10s, the predictions are averaged on 10s sub-chunks for each chunk.

Examples of datasets with randomly generated fast third-octaves are given in test.npy and test.h5.

A db offset should be specified by the user. As the PANN model is expecting normalized data as input, it is necessary 
to specify an offset to make sure that the third-octave input is in a dBFS range. We leave the choice of the 
dB offset entirely to the user. You can run the FAST-TO-EARS algorithm using a specific offset (e.g. -90) on your dataset using the 
following command:

```
python3 compute_classifier.py -n MYNAME.h5 -dbo -90
```

To have an idea of the db offset to use, we recommand using our dB offset calculation on a large dataset using the 
following command. This command will also compute the classifier on the given dataset taking the calculated offset as db offset.

```
python3 compute_classifier.py -n MYNAME.h5 -gdbo True
``` 

The results are stored in the folder "predictions", under the name predictions_MYNAME.h5, by default in a h5 format. If the output is chosen to be a h5 file, it contains a dataset named "predictions" of size `(nb_of_chunks, 527)`. To find the index of the sound class you want, please check the file `pann_classes.xlsx` which contains the correspondance between the Audioset sound sources name and the output index of PANNs.
If you want to store it as a .npy file instead of a .h5 file, please run:

```
python3 compute_classifier.py -n MYNAME.h5 -dbo -90 -of npy
```

## PANN PREDICTION FROM AUDIO

Test the PANN model on audio. Place your wav file "example.wav" in the "audio" folder and use the following code to print the top 10 predictions of PANN-Mels (PANNs using Mel spectrograms calculated from waveform audio as input) and PANN-1/3oct (PANNs using transcoded fast third-octaves as input) on the given audio. This will give you an idea of the viability of using PANNs to detect your sound source of interest.

```
python3 compute_classifier_from_audio.py example.wav
```

## REFERENCES

[1] Tailleur, M., Lagrange, M., Aumond, P., & Tourre, V. (2023, September). Spectral trancoder: using pretrained urban sound classifiers on undersampled spectral representations. In 8th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE).

[2] Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020). Panns: Large-scale pretrained audio neural networks for audio pattern recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28, 2880-2894.

[3] Gemmeke, J. F., Ellis, D. P., Freedman, D., Jansen, A., Lawrence, W., Moore, R. C., ... & Ritter, M. (2017, March). Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP) (pp. 776-780). IEEE.