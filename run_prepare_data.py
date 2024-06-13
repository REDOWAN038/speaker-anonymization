from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from shutil import copy

import pandas as pd
import torchaudio

from utils import read_kaldi_format, save_kaldi_format

LANGUAGE2TAG = {
    'dutch': 'nl',
    'english': 'en',
    'french': 'fr',
    'german': 'de',
    'italian': 'it',
    'polish': 'pl',
    'portuguese': 'pt',
    'russian': 'ru',
    'spanish': 'es'
}


def read_enrolls(filepath):
    utts = []
    with open(filepath, 'r') as f:
        for line in f:
            utts.append(line.strip())
    return utts


def read_trials(filepath):
    utts = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().split()
            utts.append(line[1])
    return utts


def utt2spk_to_spk2utt(utt2spk):
    spk2utt = defaultdict(list)
    for utt, spk in utt2spk.items():
        spk2utt[spk].append(utt)
    return spk2utt


def get_audio_dur(audiopath):
    metadata = torchaudio.info(audiopath)
    return metadata.num_frames / metadata.sample_rate


def prepare_kaldi_format(utts, split, global_utt2spk, dataset_path, out_dir, data_type='mls'):
    wav_scp = dict()
    utt2dur = dict()

    deleted_utts = []
    for utt in utts:
        if data_type == 'mls':
            mls_spk, mls_session, _ = utt.split('_')
            audio_path = dataset_path / split / 'audio' / mls_spk / mls_session / f'{utt}.flac'
        else:
            audio_path = dataset_path / 'clips' / f'{utt}.mp3'
        if not audio_path.exists():
            deleted_utts.append(utt)
            print(f'Audio does not exist: {audio_path}')
            continue
        wav_scp[utt] = str(audio_path)
        utt2dur[utt] = get_audio_dur(audio_path)

    utts = list(set(utts) - set(deleted_utts))

    if data_type == 'mls':
        text = read_kaldi_format(dataset_path / split / 'transcripts.txt', values_as_string=True)
        text = {utt: sentence for utt, sentence in text.items() if utt in utts}
    else:
        df = pd.read_csv(dataset_path / 'validated.tsv', sep='\t')
        df['path'] = df['path'].apply(lambda x: x.replace('.mp3', ''))
        df = df.set_index('path')
        text = {utt: df.loc[utt]['sentence'] for utt in utts}

    utt2spk = {utt: spk for utt, spk in global_utt2spk.items() if utt in utts}
    spk2utt = utt2spk_to_spk2utt(utt2spk)
    if data_type == 'mls':
        df = pd.read_csv(dataset_path / 'metainfo.txt', delimiter='\s+\|\s+', engine='python')
        df = df.set_index('SPEAKER')
        spk2gender = df['GENDER'].to_dict()
    else:
        spk2gender = {spk: spk[0] for spk in spk2utt.keys()}

    out_dir.mkdir(exist_ok=True, parents=True)
    save_kaldi_format(utt2spk, out_dir / 'utt2spk')
    save_kaldi_format(spk2utt, out_dir / 'spk2utt')
    save_kaldi_format(text, out_dir / 'text')
    save_kaldi_format(wav_scp, out_dir / 'wav.scp')
    save_kaldi_format(utt2dur, out_dir / 'utt2dur')
    save_kaldi_format(spk2gender, out_dir / 'spk2gender')


def prepare_data(language, dataset_path, output_path, data_type='mls'):
    trials_data_path = Path(f'trials_data/{data_type}/{language}')
    utt2spk_file = list(trials_data_path.glob('*_utt2spk'))[0]
    utt2spk = read_kaldi_format(utt2spk_file)

    for enrolls_file in trials_data_path.glob('*_enrolls'):
        utts = read_enrolls(enrolls_file)
        split = 'test' if 'test' in enrolls_file.name else 'dev'
        enroll_out_dir = output_path / enrolls_file.name
        prepare_kaldi_format(utts=utts, split=split, global_utt2spk=utt2spk, dataset_path=dataset_path,
                             out_dir=enroll_out_dir, data_type=data_type)
        copy(enrolls_file, enroll_out_dir / 'enrolls')

    for trials_file in trials_data_path.glob('*_trials*'):
        utts = read_trials(trials_file)
        split = 'test' if 'test' in trials_file.name else 'dev'
        trials_out_dir = output_path / trials_file.name
        prepare_kaldi_format(utts=utts, split=split, global_utt2spk=utt2spk, dataset_path=dataset_path,
                             out_dir=trials_out_dir, data_type=data_type)
        copy(trials_file, trials_out_dir / 'trials')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mls_path', default='MultiLingLibriSpeech')
    parser.add_argument('--cv_path', default='CommonVoice/cv-corpus-16.1-2023-12-06')
    parser.add_argument('--output_path', default='data')
    args = parser.parse_args()

    # Prepare MLS data in kaldi format based on the trials data
    languages = ['dutch', 'french', 'german', 'italian', 'portuguese', 'spanish']
    for language in languages:
        print(f'Prepare data for MLS-{language}')
        dataset_path = Path(args.mls_path, f'mls_{language}')
        prepare_data(language, dataset_path, Path(args.output_path), data_type='mls')


    # Prepare CommonVoice in kaldi format based on trials data
    languages = ['dutch', 'english', 'french', 'german', 'italian', 'polish', 'portuguese', 'russian', 'spanish']
    for language in languages:
        print(f'Prepare data for CommonVoice-{language}')
        dataset_path = Path(args.cv_path, LANGUAGE2TAG[language])
        prepare_data(language, dataset_path, Path(args.output_path), data_type='common_voice')