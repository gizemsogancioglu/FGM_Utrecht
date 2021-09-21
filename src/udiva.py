import os
import h5py
import math
import numpy as np
import pandas as pd
from file_config import *
import pickle
#import audiofile as af
#import opensmile
CULTURAL_BACKGROUND = {'Argentina': 'LATIN-AMERICAN', 'Brasil': 'LATIN-AMERICAN', 'Chile': 'LATIN-AMERICAN',
                       'Czech Republic': 'EASTERN EUROPEAN', 'Ecuador': 'LATIN-AMERICAN', 'Germany': 'GERMAN',
                       'Hungary': 'EASTERN EUROPEAN', 'Iran': 'SOUTH-EAST ASIAN', 'Kuwait': 'MIDDLE EASTERN',
                       'Mexico': 'LATIN-AMERICAN', 'Morocco': 'MIDDLE EASTERN', 'Nicaragua': 'LATIN-AMERICAN',
                       'Pakistan': 'SOUTH-EAST ASIAN', 'Peru': 'LATIN-AMERICAN', 'Romania': 'LATIN-EUROPEAN',
                       'Spain': 'LATIN-EUROPEAN', 'Syria': 'MIDDLE EASTERN', 'Uruguay': 'LATIN-AMERICAN',
                       'Venezuela': 'LATIN-AMERICAN', 'Italy': 'LATIN-EUROPEAN', 'Cuba': 'LATIN-AMERICAN', 'France': 'LATIN-EUROPEAN'}

extensive_task = {'T': 'talk', 'G': 'ghost', 'L': 'lego', 'A': 'animals'}
SESSIONS = ['SESSION%d' % d for d in range(1, 6)]
SAMPLE_DURATION = 2.5  # in seconds
#print(SESSIONS)


class UDIVA:
    def get_session_info(_set=SETS.train, root=DATA_DIR):
        # session id => {'part1': xxx, 'part2': yyy}
        sessions_df = pd.read_csv(os.path.join(root, _set, 'metadata', f'sessions_{_set}.csv'))
        return sessions_df[["ID", "PART.1", "PART.2"]]

    @staticmethod
    def get_metadata(_set=SETS.train, root=DATA_DIR):
        people_file = os.path.join(root, _set, 'metadata', f'parts_{_set}.csv')
        sessions_file = os.path.join(root, _set, 'metadata', f'sessions_{_set}.csv')

        people_df = pd.read_csv(people_file, dtype={x: str for x in SESSIONS})
        sessions_df = pd.read_csv(sessions_file)
        # gender
        df_one = pd.get_dummies(people_df["GENDER"])  # give us a column for each different label (M and F)
        df_two = pd.concat((df_one, people_df), axis=1)
        df_two = df_two.drop(["GENDER"], axis=1)
        df_two = df_two.drop(["M"], axis=1)
        people_df = df_two.rename(columns={"F": "GENDER"})

        people_df['BACKGROUND'] = people_df['COUNTRY']
        people_df.replace({'BACKGROUND': CULTURAL_BACKGROUND}, inplace=True)

        # cultural
        df_one = pd.get_dummies(people_df['BACKGROUND'])
        df_two = pd.concat((df_one, people_df), axis=1)
        people_df = df_two.drop(["BACKGROUND"], axis=1)
        people_df['BACKGROUND'] = 0
        for b, background in enumerate(sorted(list(set(CULTURAL_BACKGROUND.values())))):
            if background in people_df:
                people_df['BACKGROUND'] += people_df[background] * (b + 1)
                people_df = people_df.drop([background], axis=1)

        #print('3. ', people_df.columns)
        return people_df

    @staticmethod
    def read_annotation(path, limit=None):
        annot = {}
        with h5py.File(path, "r") as f:
            for frame_no in f.keys():
                annot[frame_no] = {}
                if not f[frame_no].attrs['valid']:
                    continue
                if 'body' not in f[frame_no] and PARTS.face not in f[frame_no] and 'hands' not in f[frame_no]:
                    continue
                if f[frame_no][PARTS.body].attrs['valid']:
                    annot[frame_no][PARTS.body] = {}
                    if 'landmarks' in f[frame_no][PARTS.body].keys():
                        annot[frame_no][PARTS.body]['landmarks'] = f[frame_no][PARTS.body]['landmarks'].__array__()
                    if 'confidence' in f[frame_no][PARTS.body].keys():
                        annot[frame_no][PARTS.body]['confidence'] = f[frame_no][PARTS.body].attrs['confidence']
                if f[frame_no][PARTS.face].attrs['valid']:
                    annot[frame_no][PARTS.face] = {}
                    if 'landmarks' in f[frame_no][PARTS.face].keys():
                        annot[frame_no][PARTS.face]['landmarks'] = f[frame_no][PARTS.face]['landmarks'].__array__()
                    if 'confidence' in f[frame_no][PARTS.face].keys():
                        annot[frame_no][PARTS.face]['confidence'] = f[frame_no][PARTS.face].attrs['confidence']
                    if 'gaze' in f[frame_no][PARTS.face].keys():
                        annot[frame_no][PARTS.face]['gaze'] = f[frame_no][PARTS.face].attrs['gaze']
                if f[frame_no]['hands']['left'].attrs['valid']:
                    annot[frame_no][PARTS.left_hand] = {}
                    if 'landmarks' in f[frame_no]['hands']['left'].keys():
                        annot[frame_no][PARTS.left_hand]['landmarks'] = \
                            f[frame_no]['hands']['left']['landmarks'].__array__()
                    if 'confidence' in f[frame_no]['hands']['left'].keys():
                        annot[frame_no][PARTS.left_hand]['confidence'] = \
                            f[frame_no]['hands']['left'].attrs['confidence']
                    annot[frame_no][PARTS.left_hand]['visible'] = f[frame_no]['hands']['left'].attrs['visible']
                if f[frame_no]['hands']['right'].attrs['valid']:
                    annot[frame_no][PARTS.right_hand] = {}
                    if 'landmarks' in f[frame_no]['hands']['right'].keys():
                        annot[frame_no][PARTS.right_hand]['landmarks'] = \
                            f[frame_no]['hands']['right']['landmarks'].__array__()
                    if 'confidence' in f[frame_no]['hands']['right'].keys():
                        annot[frame_no][PARTS.right_hand]['confidence'] = \
                            f[frame_no]['hands']['right'].attrs['confidence']
                    annot[frame_no][PARTS.right_hand]['visible'] = f[frame_no]['hands']['right'].attrs['visible']
                if limit is not None and int(frame_no) > limit:
                    break
        return annot

    @staticmethod
    def get_prediction_segments(root=DATA_DIR, session=None, view_task=None):
        path = os.path.join(root, SETS.val, 'metadata', 'validation_segments_topredict.csv')
        segments2pred = pd.read_csv(path)
        if session is None:
            return segments2pred
        else:
            segments2pred = segments2pred[segments2pred['session'] == int(session)]
            if view_task is None:
                return segments2pred
            else:
                return segments2pred[segments2pred['task'] == view_task]

    @staticmethod
    def print_annotations(_set=SETS.train, root=DATA_DIR):
        filename = os.path.join(root, _set, "annotations/002003/FC1_A/annotations_raw.hdf5")

        with h5py.File(filename, "r") as f:
            print(f"Frames: {list(f.keys())[:5]}...{list(f.keys())[-5:]}")
            a_group_key = list(f.keys())[1471]

            parts = list(f[a_group_key])
            print(f"Parts: {parts}")
            for part in parts:
                if part == 'hands':
                    print(f"{part}-left landmark counts: {len(f[a_group_key][part]['left']['landmarks'])}")
                    print(f"{part}-right landmark counts: {len(f[a_group_key][part]['right']['landmarks'])}")
                else:
                    print(f"{part} landmark counts: {len(f[a_group_key][part]['landmarks'])}")
            print(f"Body landmark examples: f{list(f[a_group_key]['body']['landmarks'])[:5]}...")
            print(f"Gaze example: f{list(f[a_group_key]['face'].attrs['gaze'])}...")

    # @staticmethod
    # def get_video_features(session):
    #     path_rgb = os.path.join(VIDEO_FEAT_DIR, f"{session}_{task}_rgb.npy")
    #     path_flow = os.path.join(VIDEO_FEAT_DIR, f"{session}_{task}_flow.npy")
    #     assert os.path.exists(path_rgb), f'{path_rgb} does not exist'
    #     assert os.path.exists(path_flow), f'{path_flow} does not exist'
    #     rgb = np.load(path_rgb)
    #     flow = np.load(path_flow)
    #     print(rgb)
    #     print(flow)

    @staticmethod

    def get_audio_features(ID, session, camera, task, mode, _set = SETS.train):

        def convert_time(timetext):
            time = timetext.split(':')
            time = float(time[1]) * 60 + float(time[2])  # in seconds
            return time

        path_lld = os.path.join(DATA_DIR, _set, 'audio', session, f"FC{camera}_{task}_ComParE_2016_LLD.csv")
        lld = []
        path_func = path_lld.replace('LLD', f'mean_std_{mode}')

        if not os.path.exists(path_func):
            lld = pd.read_csv(path_lld)
            end_lld = convert_time(lld['end'].values[-1])

            col_names = lld.columns[2:]
            mean_names = ['mean_%s' % s for s in col_names]
            std_names = ['std_%s' % s for s in col_names]
            samples = pd.DataFrame(columns = ['ID', 'session', 'camera', *mean_names, *std_names])

            path_timestamps = os.path.join(DATA_DIR, _set, 'transcriptions', session, f"{session}_{extensive_task[task]}.pkl")

            if mode == 'fixed' or not os.path.exists(path_timestamps):
                if mode == 'sentence':
                    print('Warning - session %d and camera %d are not being analysed sentence wise (transcription not found)', )
                rows_per_sample = int(SAMPLE_DURATION / convert_time(lld['start'].values[1]))
                for row in range(0, len(lld), rows_per_sample):
                    start = row
                    end = row + rows_per_sample
                    # print('start: ', start, ' end: ', end, 'len lld: ', len(lld))
                    f = lld[start:end].drop(columns=['start', 'end']).values.astype(float)

                    means = np.mean(f, axis=0)
                    stds = np.std(f, axis=0)

                    series = pd.Series([ID, session, camera, *means, *stds], index = samples.columns)
                    samples = samples.append(series, ignore_index= True)

            elif mode == 'sentence':
                path_timestamps = os.path.join(DATA_DIR, _set, 'transcriptions', session, f"{session}_{extensive_task[task]}.pkl")
                dict_ = pickle.load(open(os.path.join(DATA_DIR, _set, 'transcriptions', session, f"{session}_{extensive_task[task]}.pkl"), 'rb'))

                timestamps = dict_[f'part_{camera}']
                time_row = convert_time(lld['start'].values[1])
                for row in timestamps:

                    rows_per_sample = int((row[1] - row[0])/ time_row)
                    initial_row = int(row[0]/time_row)

                    start = initial_row
                    end = initial_row + rows_per_sample
                    f = lld[start:end].drop(columns=['start', 'end']).values.astype(float)

                    means = np.mean(f, axis=0)
                    stds = np.std(f, axis=0)

                    series = pd.Series([ID, session, camera, *means, *stds], index = samples.columns)
                    samples = samples.append(series, ignore_index= True)

            samples.to_csv(path_func, index=False)
        else:
           samples = pd.read_csv(path_func, index_col=0)
        return samples

    @staticmethod
    def get_fold(index, train, task, _set=SETS.train, root=DATA_DIR):
        path_csv = os.path.join('..', 'cross_val_division.csv')
        assert os.path.exists(path_csv), f'{path_csv} does not exist'
        csv = pd.read_csv(path_csv)

        sessions_file = os.path.join(root, _set, 'metadata', f'sessions_{_set}.csv')
        sessions_df = pd.read_csv(sessions_file, dtype={x: str for x in ['ID']})

        participants = csv[f'fold_{index}'].values.astype(int)

        y = pd.DataFrame(columns = ['participant', 'session', 'camera', *train.columns[5:10]])
        for p in participants:
            sessions = []
            p_row = train.loc[train['ID'] == p]
            for s in [1, 2, 3, 4, 5]:
                sessions.append(p_row['SESSION%d' % s].values[0])

            sessions = [x for x in sessions if str(x) != 'nan']
            for s in sessions:
                row = sessions_df.loc[sessions_df['ID'] == s]
                if row['PART.1'].values[0] == p:
                    series = pd.Series([p, s, 1, *p_row.values[0][5:10]], index = y.columns)
                    y = y.append(series, ignore_index=True)
                elif row['PART.2'].values[0] == p:
                    series = pd.Series([p, s, 2, *p_row.values[0][5:10]], index = y.columns)
                    y = y.append(series, ignore_index=True)
        return y

    def get_combined_info(train,_set=SETS.train, root=DATA_DIR):

        sessions_file = os.path.join(root, _set, 'metadata', f'sessions_{_set}.csv')
        sessions_df = pd.read_csv(sessions_file, dtype={x: str for x in ['ID']})

        y = pd.DataFrame(columns = ['participant', 'session', 'camera', *train.columns[5:10]])
        participants = train['ID'].values
        for p in participants:
            sessions = []
            p_row = train.loc[train['ID'] == p]
            for s in [1, 2, 3, 4, 5]:
                sessions.append(p_row['SESSION%d' % s].values[0])

            sessions = [x for x in sessions if str(x) != 'nan']
            for s in sessions:
                row = sessions_df.loc[sessions_df['ID'] == s]
                if row['PART.1'].values[0] == p:
                    series = pd.Series([p, s, 1, *p_row.values[0][5:10]], index = y.columns)
                    y = y.append(series, ignore_index=True)
                elif row['PART.2'].values[0] == p:
                    series = pd.Series([p, s, 2, *p_row.values[0][5:10]], index = y.columns)
                    y = y.append(series, ignore_index=True)
        return y
