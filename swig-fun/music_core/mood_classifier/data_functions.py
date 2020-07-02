import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ai_utils.blob_storage import *
from io import StringIO
import psycopg2
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

#######################################################################################################################
# Function loading data
#######################################################################################################################

def load_data_for_model_training():
    
    '''
    Run function for loading data for training mood classifier
    Returns: Two dataframes, echonest: with target data and df: music feature data 
    '''
        
    # Get echonest data from blobstorage - targets 
    blob=BlobStorage()

    blob.set_container('music')
    echonest_byte=blob.download('echonest_fma')

    s=str(echonest_byte,'utf-8')
    echonest=pd.read_csv(StringIO(s),index_col=0,header=[0,2], ) # skipinitialspace=True
    echonest=echonest.droplevel(0, axis=1) 


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
    df[['chroma_stft_01','chroma_stft_02','chroma_stft_03','chroma_stft_04','chroma_stft_05',
        'chroma_stft_06','chroma_stft_07','chroma_stft_08','chroma_stft_09','chroma_stft_10','chroma_stft_11','chroma_stft_12']] =pd.DataFrame(df.chroma_stft_mean.values.tolist(), index= df.index)
    df=df.drop('chroma_stft_mean', axis=1)

    # Unwrap spectral_contrast_mean values 
    df[['spectral_contrast_01','spectral_contrast_02','spectral_contrast_03','spectral_contrast_04','spectral_contrast_05','spectral_contrast_06','spectral_contrast_07']] = pd.DataFrame(df.spectral_contrast_mean.values.tolist(), index= df.index)
    df=df.drop('spectral_contrast_mean', axis=1)

    notIncluded=["title",'id']
    trainCols = [c for c in df.columns if c not in notIncluded]

    print('Parameteres cleaned, renamed and ready for scoring')

    return(echonest,df)

#######################################################################################################################
# Function cleaning data 
#######################################################################################################################

def clean_training_data(echonest,df):
    
    '''
    data: takes two dataframes as inputs (echonest, targets and df with music features)
    returns: X and Y array for training model 
    '''
    
    echonest=echonest[['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence']]
    targetCols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'valence'] 

    data=pd.merge(echonest,df, left_on ='track_id',right_on='id',how='inner')

    data=data.drop('title', axis=1)
    data=data.drop('id', axis=1)

    # Binarize target columns 
    data["acousticness"]=np.where(data["acousticness"]>0.5,1,0)
    data["danceability"]=np.where(data["danceability"]>0.5,1,0)
    data["energy"]=np.where(data["energy"]>0.5,1,0)
    data["instrumentalness"]=np.where(data["instrumentalness"]>0.5,1,0)
    data["valence"]=np.where(data["valence"]>0.5,1,0)
    print('        Target columns binarized')
    
    # Splitting into test and training
    xTrain, xTest = train_test_split(data, test_size = 0.3, random_state = 0)

    yTrain=list(xTrain[targetCols].values)
    xTrain= xTrain.drop(targetCols,axis = 1)
    xTrain=list(xTrain.values)

    yTest=list(xTest[targetCols].values)
    xTest= xTest.drop(targetCols,axis = 1)
    xTest=list(xTest.values)
    
    return(yTrain,xTrain)

#######################################################################################################################
# Function for fitting model
#######################################################################################################################

def fit_mood_classifier(xTrain,yTrain):
    '''
    Data: takes training data xTrain and multioutput target yTrain 
    Returns: xgb model object
    '''
    
    # Define xgb object
    XGBClassifier_object = XGBClassifier(n_estimators=4000,eta=0.01)
    xgb_mood_classifier = MultiOutputClassifier(XGBClassifier_object, n_jobs=-1)
     
    return(xgb_mood_classifier)
