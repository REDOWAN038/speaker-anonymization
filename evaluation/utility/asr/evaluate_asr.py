from pathlib import Path
import torch
from torch.utils.data import DataLoader
from speechbrain.utils.metric_stats import ErrorRateStats
import pandas as pd

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from .speechbrain_asr import InferenceSpeechBrainASR
from anonymization.modules import SpeechRecognition
from .speechbrain_asr.inference import MyDataset
from utils import read_kaldi_format, save_yaml, setup_logger

logger = setup_logger(__name__)


def evaluate_asr(eval_datasets, eval_data_dir, params, model_path, anon_data_suffix, device, backend, anon=True):
    if backend == 'speechbrain':
        return asr_eval_speechbrain(eval_datasets=eval_datasets, eval_data_dir=eval_data_dir, params=params,
                                    model_path=model_path, anon_data_suffix=anon_data_suffix, device=device)
    elif backend in ('whisper', 'wav2vec2', 'mms'):
        return asr_eval_hf(eval_datasets=eval_datasets, eval_data_dir=eval_data_dir, params=params,
                           model_path=model_path, anon_data_suffix=anon_data_suffix, device=device, anon=anon)
    else:
        raise ValueError(f'Unknown backend {backend} for ASR evaluation. Available backends: speechbrain, whisper, wav2vec2, mms.')


def asr_eval_speechbrain(eval_datasets, eval_data_dir, params, model_path, anon_data_suffix, device):
    print(f'Use ASR model for evaluation: {model_path}')
    model = InferenceSpeechBrainASR(model_path=model_path, device=device)
    results_dir = params['results_dir']
    test_sets = eval_datasets + [f'{asr_dataset}_{anon_data_suffix}' for asr_dataset in eval_datasets]
    results = []


    with torch.no_grad():
        for test_set in test_sets:
            data_path = eval_data_dir / test_set
            if (results_dir / test_set / 'wer').exists() and (results_dir / test_set / 'text').exists():
                logger.info("No WER computation  necessary; print exsiting WER results")
                references = read_kaldi_format(Path(data_path, 'text'), values_as_string=True)
                hypotheses = read_kaldi_format(Path(results_dir, test_set, 'text'), values_as_string=True)
                scores = compute_wer(ref_texts=references, hyp_texts=hypotheses, out_file=Path(results_dir,test_set, 'wer'))
            else:
                dataset = MyDataset(wav_scp_file=Path(data_path, 'wav.scp'), asr_model=model.asr_model)
                dataloader = DataLoader(dataset, batch_size=params['eval_batchsize'], shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)
                hypotheses = model.transcribe_audios(data=dataloader, out_file=Path(results_dir, test_set, 'text'))
                references = read_kaldi_format(Path(data_path, 'text'), values_as_string=True)
                scores = compute_wer(ref_texts=references, hyp_texts=hypotheses, out_file=Path(results_dir,
                                                                                     test_set, 'wer'))
            wer = scores.summarize("error_rate")
            test_set_info = test_set.split('_')
            results.append({'dataset': test_set_info[0], 'split': test_set_info[1],
                            'asr': 'anon' if 'anon' in test_set else 'original', 'WER': round(wer, 3)})
            print(f'{test_set} - WER: {wer}')
        results_df = pd.DataFrame(results)
        print(results_df)
        results_df.to_csv(results_dir / 'results.csv')
        save_yaml(params, results_dir / 'config.yaml')
        return results_df

def asr_eval_hf(eval_datasets, eval_data_dir, params, model_path, anon_data_suffix, device, anon=True):
    print(f'Use ASR model for evaluation: {model_path}')
    results_dir = params['results_dir']
    model = SpeechRecognition(devices=[device], save_intermediate=True, settings=params, force_compute=True, eval=True,
                              results_dir=results_dir)
    if anon:
        test_sets = eval_datasets + [f'{asr_dataset}_{anon_data_suffix}' for asr_dataset in eval_datasets]
    else:
        test_sets = eval_datasets
    results = []


    with torch.no_grad():
        for test_set in test_sets:
            print(test_set)
            data_path = eval_data_dir / test_set
            if (results_dir / test_set / 'wer').exists() and (results_dir / test_set / 'text').exists():
                logger.info("No WER computation  necessary; print existing WER results")
                references = read_kaldi_format(Path(data_path, 'text'), values_as_string=True)
                hypotheses = read_kaldi_format(Path(results_dir, test_set, 'text'), values_as_string=True)
                scores = compute_wer(ref_texts=references, hyp_texts=hypotheses,
                                     out_file=Path(results_dir,test_set, 'wer'))
            else:
                hypotheses = model.recognize_speech(dataset_path=data_path, dataset_name=test_set)
                references = read_kaldi_format(Path(data_path, 'text'), values_as_string=True)
                scores = compute_wer(ref_texts=references, hyp_texts=hypotheses,
                                     out_file=Path(results_dir, test_set, 'wer'))
            wer = scores.summarize("error_rate")
            test_set_info = test_set.split('_')
            results.append({'dataset': test_set_info[0], 'split': test_set_info[1],
                            'asr': 'anon' if 'anon' in test_set else 'original', 'WER': round(wer, 3)})
            print(f'{test_set} - WER: {wer}')
        results_df = pd.DataFrame(results)
        print(results_df)
        results_df.to_csv(results_dir / 'results.csv')
        save_yaml(params, results_dir / 'config.yaml')
        return results_df



def plain_text_key(path):
    tokens = []  # key: token_list
    for token in path:
        token = ''.join([t for t in token if t.isalpha() or t.isspace()])
        tokens.append(token.lower().strip().split(' '))
    return tokens

def compute_wer(ref_texts, hyp_texts, out_file):
    wer_stats = ErrorRateStats()

    ids = []
    predicted = []
    targets = []
    for utt_id, ref in ref_texts.items():
        if utt_id not in hyp_texts:   # skip the problematic samples that we skipped during inference (they were too long)
            continue
        ids.append(utt_id)
        targets.append(ref)
        predicted.append(hyp_texts[utt_id])

    wer_stats.append(ids=ids, predict=plain_text_key(predicted), target=plain_text_key(targets))
    out_file.parent.mkdir(exist_ok=True, parents=True)

    with open(out_file, 'w') as f:
        wer_stats.write_stats(f)

    return wer_stats

