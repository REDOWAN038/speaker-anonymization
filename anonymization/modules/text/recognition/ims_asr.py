import torch
torch.set_num_threads(1)
from espnet2.bin.asr_inference import Speech2Text
import soundfile
import resampy
from espnet_model_zoo.downloader import ModelDownloader, str_to_hash
from tqdm import tqdm

from utils.data_io import parse_yaml

from ..text import Text


class ImsASR:

    def __init__(self, model_path, device, ctc_weight=0.2, utt_start_token='', utt_end_token='', **kwargs):
        self.device = device
        self.model_path = model_path
        self.ctc_weight = ctc_weight
        self.utt_start_token = utt_start_token
        self.utt_end_token = utt_end_token


        # It is not sufficient to simply unzip the model.zip folder because this would not set up the environment
        # correctly. Instead, we have to call the ModelDownloader routine at least once before we can use the model.
        # However, we do not want to run this every time, so we check first if the unzipped model (stored by hash
        # value) already exists
        cache_path = model_path.parent
        d = ModelDownloader(cachedir=cache_path)
        local_url = str(model_path.absolute())
        hash = str_to_hash(local_url)

        if (cache_path / hash).exists():
            yaml = parse_yaml(cache_path / hash / 'meta.yaml')
            asr_model_file = str(cache_path / hash / yaml['files']['asr_model_file'])
            asr_train_config = str(cache_path / hash / yaml['yaml_files']['asr_train_config'])
        else:
            model_files = d.download_and_unpack(local_url)
            asr_train_config = model_files['asr_train_config']
            asr_model_file = model_files['asr_model_file']

        self.speech2text = Speech2Text(asr_train_config=asr_train_config,
                                       asr_model_file=asr_model_file,
                                       device=str(self.device),
                                       minlenratio=0.0,
                                       maxlenratio=0.0,
                                       ctc_weight=ctc_weight,
                                       beam_size=15,
                                       batch_size=1,
                                       nbest=1,
                                       quantize_asr_model=False)

        self.output = 'phones' if '-phn' in model_path.name else 'text'

    def recognize_speech_of_audio(self, audio_file):
        speech, rate = soundfile.read(audio_file)
        speech = torch.tensor(resampy.resample(speech, rate, 16000), device=self.device)

        nbests = self.speech2text(speech)
        text, *_ = nbests[0]
        text = self.utt_start_token + text + self.utt_end_token
        return text

    def recognize_speech_of_dataset(self, audio_dataset, out_dir, save_intermediate=True, job_id=None):
        texts = Text(is_phones=(self.output == 'phones'))

        if job_id is None:  # single processing
            add_suffix = None
            tqdm_params = {}
        else: # process amongst multiple processes
            add_suffix = f'_{job_id}'
            tqdm_params = {'desc': f'Job {job_id}', 'leave': True}

        i = 0
        for audio in tqdm(audio_dataset, **tqdm_params):
            utt = audio['utt']
            spk = audio['spk']
            nbests = self.speech2text(torch.tensor(audio['raw'], device=self.device))
            text, *_ = nbests[0]
            text = self.utt_start_token + text + self.utt_end_token
            texts.add_instance(sentence=text, utterance=utt, speaker=spk)

            i += 1
            if i % 100 == 0 and save_intermediate:
                texts.save_text(out_dir=out_dir, add_suffix=add_suffix)

        if save_intermediate:
            texts.save_text(out_dir=out_dir, add_suffix=add_suffix)
        return texts