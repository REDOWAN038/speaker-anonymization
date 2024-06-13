# Evaluation Files of "Probing the Feasibility of Multilingual Speaker Anonymization"

This directory contains the speaker verification trial files of the evaluation data splits for Multilingual LibriSpeech (MLS) and CommonVoice (CV) that we propose in our paper. The actual data is not included and has to be obtained separately. We do not publish a new dataset but only the files necessary to reproduce our data preparation and experiments.

## Data
All files contain the utterance IDs of the original MLS and CV corpora which makes it possible to align them to the audio files as provided by the dataset creators. If you use these datasets, you need to cite the original sources. We do not claim any rights to the audios.

We do not use the complete corpora. For MLS, "dev" and "test" correspond to the dev and test splits as provided in the MLS corpus. For CV, we divided the corpus randomly into dev and test while ensuring no speaker overlap between both splits. We use the CV 16.1 version of the corpus and restrict it to validated audios where the user specified their gender as either female or male. Further information about the data restrictions can be found in our paper.

Please note that we use the "client ID" of CV to distinguish between speakers. We acknowledge that this can lead to having the same speaker under different name multiple times in the corpus if they were assigned different client IDs. Based on our results for the original (non-anonymized) data, we believe this to have only little effect on our evaluation data.


## Data Structure
The directory contains separate folders for MLS and CV, with subfolders for each language. Each language subfolder contains 7 files:

    * dev_enrolls
    * dev_trials_f
    * dev_trials_m
    * test_enrolls
    * test_trials_f
    * test_trials_m
    * utt2spk

The file structures follow the evaluation data of the Voice Privacy Challenges (https://www.voiceprivacychallenge.org). 
The "enrolls" file contain the list of utterance IDs (i.e., audio files) that are used for enrollment of the speaker verification model. 
"trials_f" and "trials_m" correspond to the trial files for female and male speakers, respectively. Each line in a trial file consists of three constituents, separated by space: "enrollment speaker" "trial utterance" "target/nontarget". The last constituent signals whether the trial utterance was originally (i.e., before the anonymization) spoken by the enrollment speaker (target) or not (nontarget). 
The utt2spk file contains the true mapping between utterance and original speaker. This file is especially important for the CV corpus where we created new speaker names to replace the long client IDs, and where the speaker assignment is not visible in the file name.
