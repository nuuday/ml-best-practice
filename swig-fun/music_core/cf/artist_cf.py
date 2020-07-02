import pandas as pd
import numpy as np
import implicit
import scipy.sparse as sparse
import tempfile
from sqlalchemy import create_engine
import json

def main():
    plays = read_consumption_data_tmpfile(rows=7e7)
    plays_cf, model2our_master = prepare_for_cf(plays)
    sparse_artist_user = sparse.csr_matrix( (plays_cf['artist_listens'].astype(float), (plays_cf['model_id'], plays_cf['new_user_id'])) )
    
    cf_artists = implicit.als.AlternatingLeastSquares(factors=100, regularization=0.01, iterations=15, num_threads=0)
    cf_artists.fit(sparse_artist_user)
    
    model_ids = plays_cf.model_id.unique()
    similar_artists = {model2our_master.get(model_id)  : [model2our_master.get(rec) for rec,_ in cf_artists.similar_items(model_id, N=20) ] for model_id in model_ids}

    with open('similar_artists.json', 'w') as fp:
        json.dump(similar_artists, fp)


def read_consumption_data_tmpfile(rows):
    """
    Use tmpfile for faster import.
    Created table aj_consume for fast import
    """
    
    query = f"select * from aj_consume limit {rows}"

    db_engine = create_engine('postgresql://rec_engine:music_4_you@10.74.136.208:5432/music_lab_recommendation')

    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
           query=query, head="HEADER"
        )
        conn = db_engine.raw_connection()
        cur = conn.cursor()
        cur.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)
        df = df.rename(columns={"lm_id": "song_id"})
        return df


def prepare_for_cf(plays):
    """
    Builds our own master_id (this is temporary)
    Creates new ids used in cf. 
    Counts listens for artist and songs.

    Returns
     1) plays table with new model_id, user_id, song_id and song & artist listens counts. 
     2) Dict to transform from model id back to our master id
    """
    plays_table = plays.copy()

    plays_table.loc[:,"our_master"] = plays_table["master_id"].fillna(plays_table["artist_id"])

    plays_table.loc[:,'new_user_id']   = plays_table['user_id'].astype("category").cat.codes
    plays_table.loc[:,'new_song_id']   = plays_table['song_id'].astype("category").cat.codes
    plays_table.loc[:,'model_id'] = plays_table['our_master'].astype("category").cat.codes

    model2our_master = dict(zip(plays_table.model_id.astype(int), plays_table.our_master))

    plays_table.loc[:,'listens'] = plays_table.groupby(['new_user_id','new_song_id'])['new_user_id'].transform('size')
    plays_table.loc[:,'artist_listens'] = plays_table.groupby(['new_user_id','model_id'])['new_user_id'].transform('size')

    print(f"Remaing rows: {plays_table.shape[0]}")

    return plays_table, model2our_master


if __name__ == "__main__":
    main()