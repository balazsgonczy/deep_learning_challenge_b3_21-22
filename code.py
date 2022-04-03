####Important message: For the reproduceability's sake pleas USE CUDA! (The best score was achieved in colab with cuda.)

###Requirements txt library list: 

# absl-py==1.0.0
# astunparse==1.6.3
# boto3==1.21.8
# botocore==1.24.8
# cachetools==4.2.4
# catboost==1.0.3
# certifi==2021.10.8
# charset-normalizer==2.0.10
# click==8.0.3
# colorama==0.4.4
# cycler==0.11.0
# Cython==0.29.23
# docopt==0.6.2
# download==0.3.5
# flatbuffers==2.0
# fonttools==4.28.3
# gast==0.4.0
# gensim==4.1.2
# google-auth==2.3.3
# google-auth-oauthlib==0.4.6
# google-pasta==0.2.0
# graphviz==0.19
# grpcio==1.43.0
# gsdmm @ git+https://github.com/rwalk/gsdmm.git@4ad1b6b6976743681ee4976b4573463d359214ee
# h5py==3.6.0
# idna==3.3
# importlib-metadata==4.10.1
# jmespath==0.10.0
# joblib==1.1.0
# keras==2.7.0
# Keras-Preprocessing==1.1.2
# kiwisolver==1.3.2
# libclang==12.0.0
# lightgbm==3.3.1
# Markdown==3.3.6
# matplotlib==3.5.0
# nltk==3.6.5
# numpy==1.21.4
# oauthlib==3.1.1
# opt-einsum==3.3.0
# packaging==21.3
# pandas==1.3.4
# Pillow==8.4.0
# pipreqs==0.4.11
# plotly==5.4.0
# protobuf==3.19.3
# pyasn1==0.4.8
# pyasn1-modules==0.2.8
# pydub==0.25.1
# pyparsing==3.0.6
# pyphen==0.11.0
# python-dateutil==2.8.2
# pytorch-tabnet==3.1.1
# pytz==2021.3
# regex==2021.11.10
# requests==2.27.1
# requests-oauthlib==1.3.0
# rsa==4.8
# s3transfer==0.5.2
# scikit-learn==1.0.1
# scipy==1.7.3
# setuptools-scm==6.3.2
# six==1.16.0
# sklearn==0.0
# smart-open==5.2.1
# tenacity==8.0.1
# tensorboard==2.8.0
# tensorboard-data-server==0.6.1
# tensorboard-plugin-wit==1.8.1
# tensorflow==2.7.0
# tensorflow-estimator==2.7.0
# tensorflow-io-gcs-filesystem==0.23.1
# termcolor==1.1.0
# textfeatures==0.0.2
# textstat==0.7.2
# threadpoolctl==3.0.0
# tomli==1.2.2
# torch==1.10.2
# torchaudio==0.10.2
# tqdm==4.62.3
# typing_extensions==4.0.1
# urllib3==1.26.8
# Werkzeug==2.0.2
# wrapt==1.13.3
# yarg==0.1.9
# zipp==3.7.0

###Installing special libraries
#pip install download
#pip install pytorch_tabnet

###Importing libraries and setting seed
from download import download
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
import json
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

#matplotlib inline

###Set working directory
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
os.chdir(desktop)

###Training the model for clf (Tabnet) object

urltrain = "https://www.dropbox.com/s/9undp5mwff8bltn/train30k.csv?dl=0"
train30kcsv = download(urltrain, "train30k.csv", replace=True)
train = pd.read_csv("train30k.csv")

train = train.iloc[:,2:]

n_total_train = len(train)

train_indices, valid_indices = train_test_split(range(n_total_train), test_size=0.2, random_state=0)

clf = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    cat_emb_dim=1,
    lambda_sparse=1e-4, momentum=0.9, clip_value=2.,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params = {"gamma": 0.95,
                     "step_size": 20},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
)

X_train = train.iloc[:,:-1].values[train_indices]
y_train = train.iloc[:,-1].values[train_indices]

X_valid = train.iloc[:,:-1].values[valid_indices]
y_valid = train.iloc[:,-1].values[valid_indices]

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)

max_epochs = 1000 #Please only run if necessary, it might take on cuda 2-3 hours! -> It has built-in early-stopping so it won't run till the end probably.

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_name=['train', 'valid'],
    max_epochs=max_epochs, patience=100,
    batch_size=64, virtual_batch_size=64
)

###Importing test data csv
urltest = "https://www.dropbox.com/s/vugb0d3t3ucpd4t/test_10k.csv?dl=0"
test10kcsv = download(urltest, "test10k.csv", replace=True)
test_df = pd.read_csv("test10k.csv")

###Preparing preprocessed data
test = test_df.iloc[:,2:]
X_test = test.iloc[:,:-1].values

X_test = sc.transform(X_test)

###Loading model and testing test accuracy
test_pred = clf.predict(X_test)

###Generating and saving json file from predicted values
y_pred_list = test_pred.tolist()

y_pred_pds = pd.Series(y_pred_list)
 
id_pds = test_df.iloc[:,1]

pred_dict = {}
for i in range(0,len(y_pred_list)):
    value = str(y_pred_pds[i])
    key = id_pds[i]
    pred_dict[key] = value
    
with open('test.json', 'w') as f:
  json.dump(pred_dict,f)
  
print("test.json is created successfully!")

###Appendix: Generation of train and test csv tabular data (Preprocessing) - DO NOT RUN UNLESS ITS NECESSARY, IT TAKES MORE THAN 6-3 hours!
##Installing necessary libraries:

# %pip install google.colab #Only works in Google Colab!
# %pip install librosa
# %pip install torch
# %pip install numpy
# %pip install helpers
# %pip install tensorflow

# #Importing applied libraries:
# from google.colab import drive #Only works in Google Colab!
# import librosa
# import numpy as np
# import os
# import pandas as pd
# import torch
# import csv
# import tensorflow as tf

# #Connecting to drive:
# from google.colab import drive #Only works in Google Colab!
# drive.mount('/content/drive')

# #Here you can find the referencing and creation of the advanced voice extractor algorithm:
# #Source: https://pytorch.org/hub/snakers4_silero-vad_vad/, https://github.com/snakers4/silero-vad

# #Loading the VAD file
# torch.set_num_threads(1)
# model, utils = torch.hub.load(repo_or_dir='v',
#                               model='silero_vad',
#                               force_reload=True)
# (get_speech_timestamps,
#  save_audio,
#  read_audio,
#  VADIterator,
#  collect_chunks) = utils

# # Defining the function of the voice extractor to be used in the processing of the wav files
# def voice_extractor(i, sampling_rate=16000): #(sample rate set based on EDA)
#     wav = read_audio(i, sampling_rate=sampling_rate)
#     speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)

#     T = torch.empty((0))
#     for sdict in speech_timestamps:
#         T = torch.cat((T[:],wav[sdict['start']-1:sdict['end']-1]))

#     if T.nelement() == 0:
#       pass
#     else:
#       wav = T
    
#     if wav.nelement() == 0:
#       wav =  torch.empty((5))

#     only_speech_wav = tf.make_tensor_proto(wav)
#     numpy_array = tf.make_ndarray(only_speech_wav)
#     return numpy_array

# # Preprocess the WAV files into tabular csv https://librosa.org/doc/main/feature.html, Reference: https://stackoverflow.com/questions/62196212/no-backend-error-while-working-with-librosa

# create_csv_files = True

# train_csv_files = "train.csv"
# test_csv_files = "test.csv"

# def extract_wav_features(sound_files_folder, csv_file_name):
#     header = 'filename chroma_stft_mean chroma_stft_std chroma_stft_max chroma_stft_min chroma_cqt_mean chroma_cqt_std chroma_cqt_max chroma_cqt_min chroma_cens_mean chroma_cens_std chroma_cens_max chroma_cens_min melspectrogram_mean melspectrogram_std melspectrogram_max melspectrogram_min spectral_contrast_mean spectral_contrast_std spectral_contrast_max spectral_contrast_min spectral_flatness_mean spectral_flatness_std spectral_flatness_max spectral_flatness_min poly_features_mean poly_features_std poly_features_max poly_features_min tonnetz_mean tonnetz_std tonnetz_max tonnetz_min tempogram_mean tempogram_std tempogram_max tempogram_min rmse_mean rmse_std rmse_max rmse_min spectral_centroid_mean spectral_centroid_std spectral_centroid_max spectral_centroid_min spectral_bandwidth_mean spectral_bandwidth_std spectral_bandwidth_max spectral_bandwidth_min rolloff_mean rolloff_std rolloff_max rolloff_min zero_crossing_rate_mean zero_crossing_rate_std zero_crossing_rate_max zero_crossing_rate_min'
#     for i in range(1, 21):
#         header += f' mfcc{i}'
#     header += ' label'
#     header = header.split()
#     print('CSV Header: ', header)
#     file = open(csv_file_name, 'w', newline='')
#     #with file:
#     writer = csv.writer(file)
#     writer.writerow(header)
#     genres = '1 2 3 4 5 6 7 8 9 0'.split()
#     sr = 16000
#     if csv_file_name == "train.csv":
#       json_labels = y_train
#     else:
#       json_labels = y_test
#     for filename in os.listdir(sound_files_folder):
#         label = json_labels[filename]
#         number = f'{sound_files_folder}/{filename}'
#         y = voice_extractor(number)
#         chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
#         chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr) #new
#         chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr) #new
#         melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr) #new
#         spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr) #new
#         spectral_flatness = librosa.feature.spectral_flatness(y=y) #new
#         poly_features = librosa.feature.poly_features(y=y, sr=sr) #new
#         tonnetz = librosa.feature.tonnetz(y=y, sr=sr) #new
#         tempogram = librosa.feature.tempogram(y=y, sr=sr) #new
#         rmse = librosa.feature.rms(y=y)
#         spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#         rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#         zcr = librosa.feature.zero_crossing_rate(y)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr)
#         to_append = f'{filename} {np.mean(chroma_stft)} {np.std(chroma_stft)} {np.max(chroma_stft)} {np.min(chroma_stft)} {np.mean(chroma_cqt)} {np.std(chroma_cqt)} {np.max(chroma_cqt)} {np.min(chroma_cqt)} {np.mean(chroma_cens)} {np.std(chroma_cens)} {np.max(chroma_cens)} {np.min(chroma_cens)} {np.mean(melspectrogram)} {np.std(melspectrogram)} {np.max(melspectrogram)} {np.min(melspectrogram)} {np.mean(spectral_contrast)} {np.std(spectral_contrast)} {np.max(spectral_contrast)} {np.min(spectral_contrast)} {np.mean(spectral_flatness)} {np.std(spectral_flatness)} {np.max(spectral_flatness)} {np.min(spectral_flatness)} {np.mean(poly_features)} {np.std(poly_features)} {np.max(poly_features)} {np.min(poly_features)} {np.mean(tonnetz)} {np.std(tonnetz)} {np.max(tonnetz)} {np.min(tonnetz)} {np.mean(tempogram)} {np.std(tempogram)} {np.max(tempogram)} {np.min(tempogram)} {np.mean(rmse)} {np.std(rmse)} {np.max(rmse)} {np.min(rmse)} {np.mean(spec_cent)} {np.std(spec_cent)} {np.max(spec_cent)} {np.min(spec_cent)} {np.mean(spec_bw)} {np.std(spec_bw)} {np.max(spec_bw)} {np.min(spec_bw)} {np.mean(rolloff)} {np.std(rolloff)} {np.max(rolloff)} {np.min(rolloff)} {np.mean(zcr)} {np.std(zcr)} {np.max(zcr)} {np.min(zcr)}'
#         for e in mfcc:
#             to_append += f' {np.mean(e)}'
#         to_append += f' {label}'
#         writer.writerow(to_append.split())
#     file.close()
#     print("End of extract_wav_features")

# if (create_csv_files == True):
#     extract_wav_features("/content/drive/MyDrive/DL challange data/data/x/train", train_csv_files) #Change path and object to test_csv_files if you would like to generate the test csv! Otherwise it generates the train csv.
#     print("CSV file is created")
# else:
#     print("CSV files creation is skipped")

# # creating a pandas dataframe to preprocess the data

# def pre_process_data(csv_file_name):
#   data = pd.read_csv(csv_file_name)
#   return data

# df = pre_process_data(train_csv_files) #Change object to test_csv_files if you would like to generate the test csv! Otherwise it generates the train csv.
# df.to_csv(train_csv_files) #Change object to test_csv_files if you would like to generate the test csv! Otherwise it generates the train csv.