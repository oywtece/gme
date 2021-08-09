# generate ft dict for existing movie_ids

import sys
import json

# big_train_main.csv
movie_idx_csv = 2
movie_id_ft_dict = {}

for line in sys.stdin:
    # line has '\n' at the end
    line = line.strip('\n').strip('\t')
    line_split = line.split(',')

    movie_id = line_split[movie_idx_csv]
    
    if movie_id not in movie_id_ft_dict:
        # remove label
        movie_id_ft_dict[movie_id] = line_split[1:]

# save dict to file
json.dump(movie_id_ft_dict, open('movie_id_ft_dict.txt','w'))

