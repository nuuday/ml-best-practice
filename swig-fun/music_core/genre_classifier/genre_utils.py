#Import relevant librarys
import os
from io import StringIO
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from utils import *
from xgboost import XGBClassifier
import joblib
from ai_utils.blob_storage import *



#------------------------------------------ Functions for data loading -----------------------------------------------------
def load_training_data():
    '''
    Loads fma training data 
    
    Returns: Pandas dataframe with music features
    '''
    con = psycopg2.connect(database="free_user_recommendation",
                           user='rec_engine', 
                           password= 'music_4_you', 
                           host= '10.74.136.208',
                           port= '5432')

    df = pd.read_sql_query("select * from recommendations_song where sample_url != ''", con)
    return(df)

def blob_to_pandas(blob_name,blobstorage):
    '''
    Loads data from blobstorage as a byte array and convert it to a pandas dataframe 
    
    Arguments:
    ---------
    blob_name: Name of blob
    blobstorage:
    '''
    blob=blobstorage.download(blob_name)
    s=str(blob,'utf-8')
    if 'features' in blob_name:
        return(pd.read_csv(StringIO(s), index_col=0, header=[0, 1, 2]))
    if 'echonest' in blob_name:
        return(pd.read_csv(StringIO(s), index_col=0, header=[0, 1, 2]))
    if 'genres' in blob_name:
        return(pd.read_csv(StringIO(s), index_col=0))
    if 'tracks' in blob_name:
        return(pd.read_csv(StringIO(s), index_col=0, header=[0, 1]))
    
    
#------------------------------------------ Functions for data-set construction ------------------------------------------------


def create_multilabel_data(df,tracks,feature_list,genres):
    '''
    Extracts features from df into seperate columns 
    
    Arguments: 
    ---------
    df: Dataframe with training data as returned by 'load_training_data()'
    tracks: Dataframe with genre-tags
    feature_list: List of feature-names of the features in df
    
    Returns:
    -------
    train_data: Pandas dataframe with features from feature_list and targets from genres
    
   
    ''' 
    # Extract features from lists and save in combined dataframe
    dfs=[]
    for i in feature_list:
        dfs.append(pd.DataFrame(df[i].values.tolist()).rename(columns=lambda x: str(i)+'_'+str(x)))
    train_data=pd.concat(dfs, axis=1)
    
    # Set index
    train_data.set_index(train_data.id_0,inplace=True)
    
    # Merge training data with genre targets
    train_data=train_data.merge(tracks['track'][['genres','genres_all']], 
                                left_on=train_data.index, 
                                right_on=tracks['track'][['genres','genres_all']].index)
    train_data.set_index(train_data.key_0,inplace=True)
    
    # Remove unused index columns and sort by new index
    train_data.drop(columns=['key_0','id_0'],inplace=True)
    train_data.sort_index(inplace=True)
    
    # Create top_genre and subgenre columns
    train_data=train_data[train_data['genres_all']!='[]']
    train_data['genres_all']=train_data['genres_all'].apply(lambda x: x.strip('[]').split(', '))
    train_data['genres_main']=train_data['genres_all'].apply(lambda x: list(set(map(int, x)) & set(genres[genres['parent']==0].index)))
    train_data['genres_sub']=train_data[['genres_all','genres_main']].apply(lambda x: list(set(map(int, x['genres_all'])) ^ set(x['genres_main'])), axis=1)
    train_data['genres_sub']=np.where(train_data['genres_sub'].astype(str)=='[]',train_data['genres_main'],train_data['genres_sub'])
    
    
    top_groups=[4, #jazz
                9, #country
                5, # classical
                20, # Spoken
                
                13, # Easy listning
                17, # Folk
                38, # experimental
               ]
    sub_group=[79, # Reggae - dub
               130, # Europe (merge with 77,117)
               92, # african
               86, # Indian
               46, # Latin america
               102, # Middle east
               19, # funk
               53, # noise-rock
               45, # Loud-rock efter 53
               31, # metal
               25, # punk
               109, # Hardcore
               85, # Garage
               26, # Post-rock
               297, # Chip-music
               181, # Techno
               182, # house
               468, # Dubstep
              ]
    dicto = {}
    for i in genres.index:
        dicto.update({i: i})
    for i in sub_group:
        for j in list(genres[genres['parent']==i].index):
            dicto.update({j: i})
    for i in top_groups:
        for j in list(genres[genres['top_level']==i].index):
                dicto.update({j: i})
    
    train_data['genres_sub']=train_data['genres_sub'].apply(lambda x: np.unique(list(map(dicto.get, x))))
    
    return(train_data)

def binary_targets(train_data,genres):
    mlb = MultiLabelBinarizer()
    array_out=mlb.fit_transform(train_data['genres_main'])
    binary_target=pd.DataFrame(data=array_out, index=train_data['genres_main'].index,columns=mlb.classes_)
    
    return(binary_target)

def binary_sub_targets(train_data,genres):
    mlb = MultiLabelBinarizer()
    array_out=mlb.fit_transform(train_data['genres_sub'])
    binary_target=pd.DataFrame(data=array_out, index=train_data['genres_sub'].index,columns=mlb.classes_)
    
    return(binary_target)

#------------------------------------------------ training the model ------------------------------------------------
def train_model(blob):    
    genres=blob_to_pandas('genres_fma',blob)
    tracks=blob_to_pandas('tracks_fma',blob)
    fma_data=load_training_data()

    # Construct mutilabel dataset
    feature_list=['mfcc_mean','chroma_stft_mean','spectral_contrast_mean','H_mean','P_mean','tempo','id']
    multi_data=create_multilabel_data(fma_data,tracks,feature_list,genres)
    binary_sub_target=binary_sub_targets(multi_data,genres)
    
    # Remove some of the experimental songs (way to overrepresentated in fma-data compared to yousee)
    exp_samples=sample(list(binary_sub_target[(binary_sub_target['Experimental']==1)&(binary_sub_target.sum(axis=1)==1)].index),10000)
    multi_data.drop(exp_samples,inplace=True)
    binary_sub_target.drop(exp_samples,inplace=True)
    
    # Train model
    pipe = Pipeline([('oversampl', RandomOverSampler(sampling_strategy=0.5)),
                     ('undersampl', RandomUnderSampler(sampling_strategy=1)),
                     ('xgb',XGBClassifier(colsample_bytree=0.8,max_depth=200,nthread=8,n_estimators=150))
                    ])
    ovr = OneVsRestClassifier(pipe)
    ovr.fit(np.array(multi_data.drop(columns=['genres','genres_all','genres_main','genres_sub'])), binary_sub_target)
    ovr.classes_=binary_sub_target.columns
    return(ovr)

#------------------------------------------------ Other helper functions --------------------------------------------

def flatten(nested_lst):
    """ Return a list after transforming the inner lists
        so that it's a 1-D list.

    >>> flatten([[[],["a"],"a"],[["ab"],[],"abc"]])
    ['a', 'a', 'ab', 'abc']
    """
    if not isinstance(nested_lst, list):
        return(nested_lst)

    res = []
    for l in nested_lst:
        if not isinstance(l, list):
            res += [l]
        else:
            res += flatten(l)
    return(res)

#-------------------------------------------------- functions for predictions -----------------------------------------
def load_model():
    #Load model from blobstorage
    blob=BlobStorage()
    blob.set_container('music')
    blob.download_to_file(name='genre_classifier.joblib')
    with open('genre_classifier.joblib', 'rb') as fo:  
        model=joblib.load(fo)
    os.remove("genre_classifier.joblib")
    return(model)


def predict_genre(features,model,unfold=False):
    '''
    arguments:
    ---------
    features: dict with features 'mfcc', 'croma_stft', 'spectral_contrast','H','P' and 'tempo'
    
    returns:
    -------
    List of predicted genre-id's
    '''
    if (unfold==True):
        # Take only means of the features:
        data=np.concatenate((features['mfcc'][0],features['chroma_stft'][0],features['spectral_contrast'][0],features['H'][0],features['P'][0],features['tempo']),axis=0)
    
    else:
        data=features
    
    # Make predictions
    predictions=model.predict_proba(data.reshape(1, -1))
    class_names=model.classes_
    res=[]
    for pred in predictions:
        value=[class_names[i] for i in [i for i, val in enumerate(pred>0.5) if val]]
        if len(value)>0:
            res.append(value)
        else:
            res.append([class_names[i] for i in [i for i, val in enumerate(pred==np.max(pred)) if val]])
            
    return(res)

