
# coding: utf-8

# # Mood classifier 
# 
# Purpose: Determining mood in songs by developing classifier on fma data to be assigned to YouSee data 
# 
# Method: Multilabel classification 
# 
# Audio attributets - Targets:
# 
# - Acoustic - The higher the value the more acoustic the song is. Songs with high 'acousticness’ will consist mostly of natural acoustic sounds (think acoustic guitar, piano, orchestra, the unprocessed human voice), while songs with a low 'acousticness’ will consists of mostly electric sounds (think electric guitars, synthesizers, drum machines, auto-tuned vocals and so on).
# 
# - Danceability - The higher the value, the easier it is to dance to this song.
# 
# - Energy - The energy of a song - the higher the value, the more energtic song.
# 
# - Instrumentalness - The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0
# 
# - Valence - The higher the value, the more positive mood for the song. Tracks with high valence sound more positive (happy, cheerful, euphoric), while tracks with low valence sound more negative (sad, depressed, angry).
# 
# 

# In[1]:


# Import relevant libraries 
import numpy as np
import pandas as pd
import os
import psycopg2
import IPython.display as ipd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix

from xgboost import XGBClassifier

import ai_utils
from ai_utils import sql

import librosa
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import warnings

from data_functions import *

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 2000)

if not sys.warnoptions:
    warnings.simplefilter("ignore")

print('Step 1: Libraries loaded')


# ### Set up connection to music recommendation data base ('django')

# In[32]:


con = psycopg2.connect(database="free_user_recommendation",
                       user='rec_engine', 
                       password= 'music_4_you', 
                       host= '10.74.136.208',
                       port= '5432')

print("Database opened successfully")

df = pd.read_sql_query("select * from recommendations_song where sample_url is not NULL", con)
print("Data fetched successfully")
print('Df rows: ' + str(len(df)))

# Select parameteres 
keep=['id','title','mfcc_mean','tempo','chroma_stft_mean','spectral_contrast_mean']
df=df[keep]

# Unwrap mfcc_mean values 
df[['mfcc_01','mfcc_02','mfcc_03','mfcc_04','mfcc_05','mfcc_06','mfcc_07','mfcc_08']] = pd.DataFrame(df.mfcc_mean.values.tolist(), index= df.index)
df=df.drop('mfcc_mean', axis=1)

# Unwrap chroma_stft_mean values 
df[['chroma_stft_01','chroma_stft_02','chroma_stft_03','chroma_stft_04','chroma_stft_05','chroma_stft_06','chroma_stft_07','chroma_stft_08','chroma_stft_09','chroma_stft_10','chroma_stft_11','chroma_stft_12']] = pd.DataFrame(df.chroma_stft_mean.values.tolist(), index= df.index)
df=df.drop('chroma_stft_mean', axis=1)

# Unwrap spectral_contrast_mean values 
df[['spectral_contrast_01','spectral_contrast_02','spectral_contrast_03','spectral_contrast_04','spectral_contrast_05','spectral_contrast_06','spectral_contrast_07']] = pd.DataFrame(df.spectral_contrast_mean.values.tolist(), index= df.index)
df=df.drop('spectral_contrast_mean', axis=1)

notIncluded=["title",'id']
trainCols = [c for c in df.columns if c not in notIncluded]

print('Parameteres cleaned, renamed and ready for scoring')


# In[33]:


# Function for loading data files (from fma) 
def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all'),
                   ('track', 'genres_top')]
        return tracks

# Load metadata and features from fma dataset 
tracks = load('../data/tracks.csv')
genres = load('../data/genres.csv')
features = load('../data/features.csv')
echonest = load('../data/echonest.csv')
print('Step 2: Data for training model loaded')

echonest=pd.DataFrame(echonest.to_records())

# Clean up columns names to match test table
echonest.columns = echonest.columns.str.replace("(", "")
echonest.columns = echonest.columns.str.replace(")", "")
echonest.columns = echonest.columns.str.replace("'", "")
echonest.columns = echonest.columns.str.replace("audio_features", "")
echonest.columns = echonest.columns.str.replace("echonest", "")
echonest.columns = echonest.columns.str.replace(", , ", "")

keep_echonest=['track_id','acousticness', 'danceability', 'energy', 'instrumentalness', 'valence']
echonest=echonest[keep_echonest]
print('        Target data ready')


# In[34]:


data=pd.merge(echonest,df, left_on ='track_id',right_on='id',how='inner')
print(len(data))

targetCols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence'] # dropping cols as they have proven very hard to predict correctly 

# Binarize target columns 
data["acousticness"]=np.where(data["acousticness"]>0.5,1,0)
data["danceability"]=np.where(data["danceability"]>0.5,1,0)
data["energy"]=np.where(data["energy"]>0.5,1,0)
data["instrumentalness"]=np.where(data["instrumentalness"]>0.5,1,0)
data["valence"]=np.where(data["valence"]>0.5,1,0)
print('        Target columns binarized')

data=data.drop('track_id', axis=1)
data=data.drop('title', axis=1)
data=data.drop('id', axis=1)

# Clean data if any missing values
print('Missing Values:')
print(data.isnull().sum().any())

# Splitting into test and training
xTrain, xTest = train_test_split(data, test_size = 0.3, random_state = 0)
yTrain=xTrain[targetCols].values
xTrain= xTrain.drop(targetCols,axis = 1)
yTest=xTest[targetCols].values
xTest= xTest.drop(targetCols,axis = 1)
print('        Data split into test and training')


# ### Binary targets

# In[20]:


# # Check if some of the mood variables are not represented at all - in that case, drop them
# cols = data_bin[targetCols].columns
# counts = {}
# for c in cols:
#     val = data_bin[c].sum()
#     counts[c] = val
#     if val == 0:
#         data_bin.drop(c, axis = 1, inplace = True)
# print('Target - counts:')        
# print(counts)


# ### Train XGBoost model 

# In[35]:


# Define xgb object
XGBClassifier_object = XGBClassifier(n_estimators=4000,eta=0.01)
xgb_classifier = MultiOutputClassifier(XGBClassifier_object, n_jobs=-1)

# Fit and predict
yScore=xgb_classifier.fit(xTrain, yTrain).predict(xTest)


# In[38]:


# Get feature importance of each xgb classifiers and take the average of them all together to get each features average importance across all classifiers 
feat_impts = [] 

for clf in xgb_classifier.estimators_:
    feat_impts.append(clf.feature_importances_)
    
meanvalues = np.mean(feat_impts, axis=0)

feature_importance=pd.DataFrame([meanvalues],columns=['tempo','mfcc_01', 'mfcc_02', 'mfcc_03', 'mfcc_04',
       'mfcc_05', 'mfcc_06', 'mfcc_07', 'mfcc_08', 'chroma_stft_01',
       'chroma_stft_02', 'chroma_stft_03', 'chroma_stft_04', 'chroma_stft_05',
       'chroma_stft_06', 'chroma_stft_07', 'chroma_stft_08', 'chroma_stft_09',
       'chroma_stft_10', 'chroma_stft_11', 'chroma_stft_12',
       'spectral_contrast_01', 'spectral_contrast_02', 'spectral_contrast_03',
       'spectral_contrast_04', 'spectral_contrast_05', 'spectral_contrast_06',
       'spectral_contrast_07']) 


# In[39]:


# Sort meanvalues
sort_indices = np.argsort(meanvalues)[::-1]
meanvalues[:] = meanvalues[sort_indices]


# In[40]:


meanvalues


# In[41]:


plt.plot(meanvalues,)
plt.show()


# In[42]:


y_true = yTest
y_pred =yScore


# ### Accuracy 

# In[43]:


print(accuracy_score(y_true, y_pred))


# ### Confusion matrix 

# In[44]:


multilabel_confusion_matrix(y_true, y_pred)


# In[64]:


n_classes=5

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# ### Test multi label classifier on fma data

# In[45]:


# Test model on YouSee data 
yScore = xgb_classifier.predict(data[trainCols])


# In[46]:


# Add predicted labels to music df

predictions=pd.DataFrame(yScore, columns=['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence'])
df_predictions=pd.merge(df,predictions, left_index=True, right_index=True)
len(df_predictions)


# In[47]:


keep=['id', 'title', 'tempo', 'acousticness', 'danceability', 'energy',
       'instrumentalness', 'valence']
df_predictions_stripped=df_predictions[keep]


# In[48]:


df_predictions_stripped.tail(20)


# ### Multilabel regression model 

# In[41]:


data.head()


# In[78]:


XGBRegression_object = XGBRegressor(n_estimators=1000,eta=0.01)
Reg_classifier = MultiOutputRegressor(XGBRegression_object, n_jobs=-1)
Reg_classifier.fit(xTrain, yTrain)

yScore = Reg_classifier.predict(xTest)

y_true = yTest
y_pred = yScore


# In[79]:


n_classes=5

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# In[49]:


# Add predicted labels to music df

predictions=pd.DataFrame(yScore, columns=['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence'])
df_predictions=pd.merge(df,predictions, left_index=True, right_index=True)
len(df_predictions)

keep=['id', 'title', 'tempo', 'acousticness', 'danceability', 'energy',
       'instrumentalness', 'valence']
df_predictions_stripped=df_predictions[keep]


# In[51]:


df_predictions_stripped.head(30)


# ### Get YouSee data from API

# In[49]:


print('Step 1: Import unique song IDs from SQL')

SQL_SERVER_HOST = '10.74.133.90'
SQL_SERVER_PORT = '2501'
SQL_SERVER_DB   = 'DM'
SQL_SERVER_UID  = 'accdom01\R02662'
SQL_SERVER_PWD  = 'vinter04'

q = """select distinct top 100  [LMID] FROM [DM].[PA].[Musik_week_22] where ActivityType = 'PLAYCOUNT' """

# From ai_utils.sql import create_engine
engine = ai_utils.sql.create_engine(SQL_SERVER_HOST, SQL_SERVER_PORT, SQL_SERVER_DB, SQL_SERVER_UID, SQL_SERVER_PWD)

song_ids = pd.read_sql(q, engine)

print('        Import done')

# Create list of unique song_ids
id_list = list(song_ids['LMID'])

# Create list of features we want to extract - (features from librosa)
features_list = ['mfcc','spectral_contrast','chroma_stft']

print('        Features_list ready (features from Librosa to be extracted from following function)')


# ### First extract the audio vector from mp3_load_util.py script

# In[ ]:


features_list=pd.Series(features_list)
features_list= np.array(features_list)
print(features_list)

id_list = pd.Series(id_list)
id_list=id_list.astype(int)


# Imported from music core to extract librosa features som YouSee data 
from data_utils.mp3_load_util import *
from music_data_gen.music_procces import *

df_ys = music_data_generator(id_list, features_list, save=False)


# In[ ]:


df_ys.head()


# In[ ]:


# Unwrap chroma_stft_mean values 
df_ys[['chroma_stft_01','chroma_stft_02','chroma_stft_03','chroma_stft_04','chroma_stft_05','chroma_stft_06','chroma_stft_07','chroma_stft_08','chroma_stft_09',
    'chroma_stft_10','chroma_stft_11','chroma_stft_12','spectral_contrast_01','spectral_contrast_02','spectral_contrast_03','spectral_contrast_04','spectral_contrast_05',
    'spectral_contrast_06','spectral_contrast_07','mfcc_01','mfcc_02','mfcc_03','mfcc_04','mfcc_05','mfcc_06','mfcc_07','mfcc_08']] = pd.DataFrame(df_ys.means_vector.values.tolist(), index= df_ys.index)
df_ys=df_ys.drop('means_vector', axis=1)
df_ys=df_ys.drop('feature_order', axis=1)


# In[ ]:


df_ys.head()


# In[ ]:


artist.head()


# In[62]:


# Test model on YouSee data 
yScore = xgb_classifier.predict(df_ys[trainCols])


# In[64]:


# Add predicted labels to music df

predictions=pd.DataFrame(yScore, columns=['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence'])
df_ys_predictions=pd.merge(df_ys,predictions, left_index=True, right_index=True)
len(df_ys_predictions)

keep=['id', 'title', 'tempo', 'acousticness', 'danceability', 'energy',
       'instrumentalness', 'valence']
df_ys_predictions_stripped=df_ys_predictions[keep]

