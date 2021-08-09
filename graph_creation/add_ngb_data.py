# add ngb fts after ori fts
# output csv data
# cre_id -> [ngb_dict] ngb -> [ft_dict] fts

import sys
import json

ft_dict_fn = sys.argv[1] # ft dict
ngb_dict_fn = sys.argv[2] # ngb dict

movie_idx_csv = 2
max_n_ngb = 10
n_ngb_ft = 16 # num_csv_col - 1 [no label]

movie_id_ft_dict = json.load(open(ft_dict_fn)) # meta_ori_movie_id_ft_dict.txt
movie_id_ngb_dict = json.load(open(ngb_dict_fn)) # train_oneshot_a_movie_id_ngb_dict.txt

for line in sys.stdin:
    # line has '\n' at the end
    line = line.strip('\n').strip('\t')
    line_split = line.split(',')

    movie_id = line_split[movie_idx_csv]
    
    ft_str = ''
    
    # if have ngb
    if movie_id in movie_id_ngb_dict:
        cur_ngb_movie_id_list = movie_id_ngb_dict[movie_id]
        cur_max_len = min(len(cur_ngb_movie_id_list), max_n_ngb)
        for i in range(cur_max_len):
            # key
            cur_ngb_movie_id = cur_ngb_movie_id_list[i]
            # val (fts)
            cur_ngb_ft = ','.join(movie_id_ft_dict[cur_ngb_movie_id])
            
            ft_str += cur_ngb_ft + ','
            
        n_remain_ngb = max_n_ngb - cur_max_len
        if n_remain_ngb > 0:
            ft_str += '0,'*n_ngb_ft*n_remain_ngb
        # remove the last ,
        ft_str = ft_str.strip(',')
    else:
        ft_str += '0,'*n_ngb_ft*max_n_ngb
        ft_str = ft_str.strip(',')
    
    print(line + ',' + ft_str)

