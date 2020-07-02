import os
import fasttext

# Save temp file from binary and then use it to load into fasttext
# Fasttext need to be loaded - Specifically the load_module() func
# os lib also needs to be loaded
def load_model_from_binary(binary_data):
        temp_path = str(os.getcwd()+'/tmp_file_model.bin')
        temp_bin_file = open(temp_path, 'wb')
        temp_bin_file.write(binary_data)
        temp_bin_file.close()
        model = fasttext.load_model(temp_path)
        os.remove(temp_path)
        return(model)

def predict_get_language(model, song_title):
    
    if song_title == "" or song_title.isdigit() or song_title.isspace():
        return(None)
    
    _prediction_temp = model.predict(song_title, k=3)[0]
    _prediction_list = [x.split('__label__')[1] for x in _prediction_temp]
    
    if _prediction_list is not None:
        return(_prediction_list)
    else:
        return(None)



