# Import libraries
import os
os.environ['OMP_NUM_THREADS'] = '8'
from genre_utils import *
from ai_utils.blob_storage import *
import joblib

blob=BlobStorage()
blob.set_container('music')

#Train model
genre_classifer=train_model(blob)

 # save model as pickle object
with open('genre_classifier.joblib', 'wb') as fo:  
    joblib.dump(genre_classifer, fo)
blob.upload_from_file('genre_classifier.joblib')
os.remove("genre_classifier.joblib")
    
