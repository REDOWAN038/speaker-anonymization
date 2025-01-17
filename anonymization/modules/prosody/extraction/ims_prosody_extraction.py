import torch
torch.set_num_threads(1)

from torch.optim import SGD
from tqdm import tqdm
import soundfile as sf

from anonymization.modules.tts.IMSToucan.Preprocessing.AudioPreprocessor import AudioPreprocessor
from anonymization.modules.tts.IMSToucan.Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from anonymization.modules.tts.IMSToucan.Preprocessing.articulatory_features import get_feature_to_index_lookup
from anonymization.modules.tts.IMSToucan.TrainingInterfaces.Text_to_Spectrogram.AutoAligner.Aligner import Aligner
from anonymization.modules.tts.IMSToucan.TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.DurationCalculator import DurationCalculator
from anonymization.modules.tts.IMSToucan.TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.EnergyCalculator import EnergyCalculator
from anonymization.modules.tts.IMSToucan.TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.PitchCalculator import Parselmouth

from utils import setup_logger

logger = setup_logger(__name__)

class ImsProsodyExtractor:

    def __init__(self, aligner_path, device, on_line_fine_tune=True, random_offset_lower=None,
                 random_offset_higher=None, language='en'):
        self.on_line_fine_tune = on_line_fine_tune
        self.random_offset_lower = random_offset_lower
        self.random_offset_higher = random_offset_higher

        self.ap = AudioPreprocessor(input_sr=16000, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        self.tf = ArticulatoryCombinedTextFrontend(language=language)
        self.device = device
        self.acoustic_model = Aligner()
        self.aligner_weights = torch.load(aligner_path, map_location='cpu')["asr_model"]
        self.acoustic_model.load_state_dict(self.aligner_weights)
        self.acoustic_model = self.acoustic_model.to(self.device)
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # torch 1.9 has a bug in the hub loading, this is a workaround
        # careful: assumes 16kHz or 8kHz audio
        self.silero_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                  model='silero_vad',
                                                  force_reload=False,
                                                  onnx=False,
                                                  verbose=False)
        (self.get_speech_timestamps, _, _, _, _) = utils
        torch.set_grad_enabled(True)  # finding this issue was very infuriating: silero sets
        # this to false globally during model loading rather than using inference mode or no_grad
        self.parsel = Parselmouth(reduction_factor=1, fs=16000)
        self.energy_calc = EnergyCalculator(reduction_factor=1, fs=16000)
        self.dc = DurationCalculator(reduction_factor=1)

    def extract_prosody(self, transcript, ref_audio_path, input_is_phones=False):
        # remove special characters
        transcript = transcript.replace('«', ''). replace('»', '')
        # we need to reinitialize the aligner weights for each utterance again if we fine tune it for each utterance
        if self.on_line_fine_tune:
            self.acoustic_model.load_state_dict(self.aligner_weights)

        wave, sr = sf.read(ref_audio_path)
        # if self.tf.language != lang:
        #     self.tf = ArticulatoryCombinedTextFrontend(language=lang)
        #logger.info(f'tf.lang: {self.tf.language}')
        if self.ap.sr != sr:
            self.ap = AudioPreprocessor(input_sr=sr, output_sr=16000, melspec_buckets=80, hop_length=256, n_fft=1024, cut_silence=False)
        try:
            norm_wave = self.ap.audio_to_wave_tensor(normalize=True, audio=wave)
        except ValueError:
            logger.error('Something went wrong, the reference wave might be too short.')
            raise RuntimeError

        with torch.inference_mode():
            speech_timestamps = self.get_speech_timestamps(norm_wave, self.silero_model, sampling_rate=16000)
        start_silence = speech_timestamps[0]['start']
        end_silence = len(norm_wave) - speech_timestamps[-1]['end']
        norm_wave = norm_wave[speech_timestamps[0]['start']:speech_timestamps[-1]['end']]

        norm_wave_length = torch.LongTensor([len(norm_wave)])
        text = self.tf.string_to_tensor(transcript, handle_missing=True, input_phonemes=input_is_phones).squeeze(
            0)
        melspec = self.ap.audio_to_mel_spec_tensor(audio=norm_wave, normalize=False, explicit_sampling_rate=16000).transpose(0, 1)
        melspec_length = torch.LongTensor([len(melspec)]).numpy()

        if self.on_line_fine_tune:
            # we fine-tune the aligner for a couple steps using SGD. This makes cloning pretty slow, but the results are greatly improved.
            steps = 4
            tokens = self.tf.text_vectors_to_id_sequence(
                text_vector=text)  # we need an ID sequence for training rather than a sequence of phonological features
            tokens = torch.LongTensor(tokens).squeeze().to(self.device)
            tokens_len = torch.LongTensor([len(tokens)]).to(self.device)
            mel = melspec.unsqueeze(0).to(self.device)
            #mel.requires_grad = True
            mel_len = torch.LongTensor([len(mel[0])]).to(self.device)
            # actual fine-tuning starts here
            optim_asr = torch.optim.Adam(self.acoustic_model.parameters(), lr=0.00001) #SGD(self.acoustic_model.parameters(), lr=0.1)
            self.acoustic_model.train()
            for _ in list(range(steps)):
                pred = self.acoustic_model(mel.clone())
                loss = self.acoustic_model.ctc_loss(pred.transpose(0, 1).log_softmax(2), tokens, mel_len, tokens_len)
                optim_asr.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.acoustic_model.parameters(), 1.0)
                optim_asr.step()
            self.acoustic_model.eval()

        # We deal with the word boundaries by having 2 versions of text: with and without word boundaries.
        # We note the index of word boundaries and insert durations of 0 afterwards
        text_without_word_boundaries = list()
        indexes_of_word_boundaries = list()
        for phoneme_index, vector in enumerate(text):
            if vector[get_feature_to_index_lookup()["word-boundary"]] == 0:
                text_without_word_boundaries.append(vector.numpy().tolist())
            else:
                indexes_of_word_boundaries.append(phoneme_index)
        matrix_without_word_boundaries = torch.Tensor(text_without_word_boundaries)

        alignment_path = self.acoustic_model.inference(mel=melspec.to(self.device),
                                                       tokens=matrix_without_word_boundaries.to(self.device),
                                                       return_ctc=False)

        duration = self.dc(torch.LongTensor(alignment_path), vis=None).cpu()

        for index_of_word_boundary in indexes_of_word_boundaries:
            duration = torch.cat([duration[:index_of_word_boundary],
                                  torch.LongTensor([0]),  # insert a 0 duration wherever there is a word boundary
                                  duration[index_of_word_boundary:]])

        # last_vec = None
        # for phoneme_index, vec in enumerate(text):
        #     if last_vec is not None:
        #         if last_vec.numpy().tolist() == vec.numpy().tolist():
        #             # we found a case of repeating phonemes!
        #             # now we must repair their durations by giving the first one 3/5 of their sum and the second one 2/5 (i.e. the rest)
        #             dur_1 = duration[phoneme_index - 1]
        #             dur_2 = duration[phoneme_index]
        #             total_dur = dur_1 + dur_2
        #             new_dur_1 = int((total_dur / 5) * 3)
        #             new_dur_2 = total_dur - new_dur_1
        #             duration[phoneme_index - 1] = new_dur_1
        #             duration[phoneme_index] = new_dur_2
        #     last_vec = vec

        # with torch.inference_mode():
        energy = self.energy_calc(input_waves=norm_wave.unsqueeze(0),
                                  input_waves_lengths=norm_wave_length,
                                  feats_lengths=melspec_length,
                                  text=text,
                                  durations=duration.unsqueeze(0),
                                  durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()

        pitch = self.parsel(input_waves=norm_wave.unsqueeze(0),
                            input_waves_lengths=norm_wave_length,
                            feats_lengths=melspec_length,
                            text=text,
                            durations=duration.unsqueeze(0),
                            durations_lengths=torch.LongTensor([len(duration)]))[0].squeeze(0).cpu()

        return duration, pitch, energy, start_silence, end_silence
