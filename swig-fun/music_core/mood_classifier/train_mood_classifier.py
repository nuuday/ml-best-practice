from data_functions import *
from ai_utils.blob_storage import *

# Import data, train model and return model object
echonest,df=load_data_for_model_training()
yTrain,xTrain=clean_training_data(echonest,df)
xgb_mood_classifier=fit_mood_classifier(xTrain,yTrain)

# # Dump model in blobStorage 
# model_object=xgb_mood_classifier
# container_name='music'
# model_name=xgb_mood_classifier

# blob=BlobStorage()
# blob.set_container(container_name)
# blob.upload(pickle.dumps(model_object), model_name)
