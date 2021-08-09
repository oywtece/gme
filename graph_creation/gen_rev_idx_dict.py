# get title:set(title), genre:set(genre), year:set(year)
# input: csv data

# 0-label, 1-user_id, 2-movie_id, 3-gender, 4-age, 
# 5-occ, 6-year, 7-title (mul), 8-genres (mul)

import sys
import json

# big_train_main.csv
movie_idx_csv = 2
year_idx_csv = 6
title_start_idx_csv = 7
title_end_idx_csv = 12
genre_start_idx_csv = 12
genre_end_idx_csv = 17

year_dict = {}
title_dict = {}
genre_dict = {}

for line in sys.stdin:
    # line has '\n' at the end
    line = line.strip('\n').strip('\t')
    line_split = line.split(',')
    
    movie_id = line_split[movie_idx_csv]
    year = line_split[year_idx_csv]
    title = line_split[title_start_idx_csv:title_end_idx_csv]
    genre = line_split[genre_start_idx_csv:genre_end_idx_csv]
    
    if year not in year_dict:
        year_dict[year] = set()
    year_dict[year].add(movie_id)
    
    for item in title:
        if item != '0':
            if item not in title_dict:
                title_dict[item] = set()
            title_dict[item].add(movie_id)
    
    for item in genre:
        if item != '0':
            if item not in genre_dict:
                genre_dict[item] = set()
            genre_dict[item].add(movie_id)
        
###########################
# set to list [set is not JSON serializable]
for key in year_dict:
    year_dict[key] = list(year_dict[key])
for key in title_dict:
    title_dict[key] = list(title_dict[key])
for key in genre_dict:
    genre_dict[key] = list(genre_dict[key])

# save dict to file
json.dump(year_dict, open('year_dict.txt','w'))
json.dump(title_dict, open('title_dict.txt','w'))
json.dump(genre_dict, open('genre_dict.txt','w'))

