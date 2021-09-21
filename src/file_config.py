import os
DATA_DIR = '../data'

FEAT_DIR = os.path.join(DATA_DIR, 'features')
VISUAL_FEAT_DIR = os.path.join(FEAT_DIR, 'visual')

# video features are in this directory with names {session}_{task}_rgb.npy and {session}_{task}_flow.npy
VIDEO_FEAT_DIR = os.path.join(VISUAL_FEAT_DIR, 'i3d')

# @Francisca: Could you put the audio features into this directory?
# AUDIO_FEAT_DIR = os.path.join(FEAT_DIR, 'audio')
# audio features are in this directory with names {task}_ComParE_2016_LLD.npy
AUDIO_FEAT_DIR = os.path.join(DATA_DIR, 'audio')

LING_FEAT_DIR = os.path.join(DATA_DIR, 'ling')
# output results will be written to the EXPERIMENTS_DIR
EXPERIMENTS_DIR = os.path.join(DATA_DIR, 'experiments')

# repository to evaluate submission files: https://github.com/crisie/UDIVA.git
UDIVA_EVAL_DIR = '/mnt/hdd1/GithubRepos/UDIVA'


# res/predictions.json, ref/ground_truth.json, and aux/distances.csv are in this folder
FORECAST_OUTPUT_EXAMPLE_DIR = os.path.join(UDIVA_EVAL_DIR, 'DYAD@ICCV21', 'behavior_forecasting', 'input')

# ref/ground_truth_sample_valid.json, and res/submission_sample_valid.csv are in this folder
PERSONALITY_OUTPUT_EXAMPLE_DIR = os.path.join(UDIVA_EVAL_DIR, 'DYAD@ICCV21', 'personality_recognition', 'input')


class SETS:
    train = 'train'
    val = 'val'
    test = 'test'


class TASKS:
    animal = 'A'
    ghost = 'G'
    talk = 'T'
    lego = 'L'


class PARTS:
    face = 'face'
    body = 'body'
    left_hand = 'left_hand'
    right_hand = 'right_hand'


class TRACKS:
    forecasting = 'behavior_forecasting'
    personality = 'personality_recognition'


class BASELINE:
    static = 'static'
    mirrored = 'mirrored'