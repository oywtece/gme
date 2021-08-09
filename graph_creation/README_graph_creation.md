# Graph Creation
The following examples are based on the MovieLens-1M dataset (check the **data** folder). \
You need to copy the scripts to the data folder in order to run them.

### Build graph according to "movie_id"

0-label, 1-user_id, 2-movie_id, 3-gender, 4-age, 5-occ, 6-year, 7-title (mul), 8-genres (mul)
 
## 1) Generate Attribute-ID reverse index dict (based on old ads)
* input: big_train_main.csv
* output (json dicts): \
year_dict.txt: key: year, val: movie_id (movie_id which was released in a corresponding year) \
title_dict.txt: key: title word, val: movie_id \
genre_dict.txt: key: genre, val: movie_id
* python script: gen_rev_idx_dict.py
* run \
$ nohup cat big_train_main.csv | python gen_rev_idx_dict.py

## 2) Generate ft dict for movie_id in big_train_main.csv
* input: big_train_main.csv
* output (json dict): \
movie_id_ft_dict.txt: key: movie_id, val: ori fts 

**Note: movie fts are selected inside models (acc. to config); not in graph creation**

* python script: gen_ft_dict.py
* run \
$ nohup cat big_train_main.csv | python gen_ft_dict.py

## 3) Get ngbs (both for old ads [for training] and new ads [for testing])
* input: train_oneshot_a.csv
* output : train_oneshot_a_movie_id_ngb_dict.txt \
key: a movie_id in train_oneshot_a.csv, val: 10 graph ngbs (movie_ids)
* python script: get_ngb.py
* run \
$ nohup bash run_get_ngb.sh

## 4) Insert ngb fts
* input: train_oneshot_a.csv
* output: train_oneshot_a_w_ngb.csv
* python script: add_ngb_data.py
* run \
$ nohup bash run_add_ngb_data.sh
