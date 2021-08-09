# input dict: title:set(movie_id), genre:set(movie_id), year:set(movie_id)
# output dict: movie_id:set(movie_id)

# 0-label, 1-user_id, 2-movie_id, 3-gender, 4-age, 
# 5-occ, 6-year, 7-title (mul), 8-genres (mul)

import sys
import json

tar_fn = sys.argv[1]

max_n_ngb = 10

# big_train_main.csv
movie_idx_csv = 2
year_idx_csv = 6
title_start_idx_csv = 7
title_end_idx_csv = 12
genre_start_idx_csv = 12
genre_end_idx_csv = 17

# load dict from file
year_dict = json.load(open('year_dict.txt'))
title_dict = json.load(open('title_dict.txt'))
genre_dict = json.load(open('genre_dict.txt'))

movie_id_ngb_dict = {}

with open(tar_fn + '.csv', 'rt') as f:
    for line in f:
        line = line.strip('\n').strip('\t')
        line_split = line.split(',')
        
        movie_id = line_split[movie_idx_csv]
        
        if movie_id not in movie_id_ngb_dict:                    
            year = line_split[year_idx_csv]
            title = line_split[title_start_idx_csv:title_end_idx_csv]
            genre = line_split[genre_start_idx_csv:genre_end_idx_csv]
    
            ngb_from_year = set()
            if year in year_dict:
                ngb_from_year = set(year_dict[year])
            
            ngb_from_title = set()
            for item in title:
                # '0' is not in title_dict
                if item in title_dict:
                    ngb_from_title = ngb_from_title.union(set(title_dict[item]))
            
            ngb_from_genre = set()
            for item in genre:
                if item in genre_dict:
                    ngb_from_genre = ngb_from_genre.union(set(genre_dict[item]))
            
            ngb_set = ngb_from_year.union(ngb_from_title, ngb_from_genre)
            
            ## rank ngbs
            ngb_score_dict = {}
            
            for cur_ngb in ngb_set:
                # remove itself
                if cur_ngb != movie_id:
                    score = 0
                    if cur_ngb in ngb_from_year:
                        score += 1
                    for item in title:
                        if item != '0':
                            if item in title_dict and cur_ngb in set(title_dict[item]):
                                score += 1
                    for item in genre:
                        if item != '0':
                            if item in genre_dict and cur_ngb in set(genre_dict[item]):
                                score += 1
                    ngb_score_dict[cur_ngb] = score
            
            # sort dict by val
            ngb_score_dict_sorted_keys = sorted(ngb_score_dict, key=ngb_score_dict.get, reverse=True)
#             print(cre_id)
#             for key in ngb_score_dict_sorted_keys:
#                 print(key, ngb_score_dict[key])
            
            if len(ngb_score_dict_sorted_keys) > max_n_ngb:
                ngb_list = ngb_score_dict_sorted_keys[:max_n_ngb]
            else:
                ngb_list = ngb_score_dict_sorted_keys
            movie_id_ngb_dict[movie_id] = ngb_list
            
# save dict to file
json.dump(movie_id_ngb_dict, open(tar_fn + '_movie_id_ngb_dict.txt','w'))

