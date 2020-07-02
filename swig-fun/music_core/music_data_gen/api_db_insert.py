"""God damit i hate docstrings."""
from __future__ import generators
import psycopg2 as psy
import requests
from src.music_data_gen.rarus_music_process import process_wrapper as pw


def get_sql_itr():
    """Script for uploading data to database via api."""
    def _sql_iter(cursor, arraysize=1000):

        while True:
            results = cursor.fetchmany(arraysize)
            if not results:
                break
            for result in results:
                yield result

    query = ''' SELECT id, lmid FROM recommendations_song where sample_url is not null and sample_url not LIKE '' '''
    connection = psy.connect(
        user="rec_engine", password="music_4_you", host="10.74.136.208",
        port="5432",
        database="music_recommendation")

    cursor = connection.cursor()
    cursor.execute(query)
    return(cursor)


API_ENDPOINT = 'http://127.0.0.1:8000/api/song/{id}/'
LOGIN_URL = 'http://127.0.0.1:8000/accounts/login/'

api_header = {
    'content-type': 'application/json',
    'Authorization': 'ApiKey data_feed:88277a6d75085194cc66b91e21492916b6da5dd2'}

# THE GREAT LOOP OF UGLY CODE
itr = 0
for row in get_sql_itr():
    if itr < 60000:
        id = int(row[0])
        lmid = int(row[1])
        temp_for_dict = pw(lmid)
        if temp_for_dict is not None:
            dict_to_post = {
                        "tempo": int(temp_for_dict[0]),
                        "cov_matrix": temp_for_dict[1].astype('double').tolist(),
                        "feature_means": [float(el) for el in temp_for_dict[2]],
                        "feature_order": temp_for_dict[3]}

            ra = requests.put(
                API_ENDPOINT.format(type='song', id=id),
                json=dict_to_post,
                headers=api_header)
            print("{0}: {1} ->{2}".format(itr, lmid, ra.status_code))
        else:
            print("MP3 Could not be loaded")
    itr += 1
