'''
config file
'''

n_one_hot_slot = 6 # 0 - user_id,  1 - movie_id, 2 - gender, 3 - age, 4 - occ, 5 - release year
n_mul_hot_slot = 2 # 6 - title (mul-hot), 7 - genres (mul-hot)
max_len_per_slot = 5 # max num of fts in one mul-hot slot
num_csv_col_warm = 17
num_csv_col_w_ngb = 17 + 160 # num of cols in the csv file (w ngb)
layer_dim = [256, 128, 1]

# for ngb
n_one_hot_slot_ngb = 6
n_mul_hot_slot_ngb = 2
max_len_per_slot_ngb = 5
max_n_ngb_ori = 10 # num of ngbs in data file
max_n_ngb = 10 # num of ngbs to use in model, <= max_n_ngb_ori

pre = './data/'
suf = '.tfrecord'

# a, b - used for meta learning
train_file_name_a = [pre+'train_oneshot_a_w_ngb'+suf, pre+'train_oneshot_b_w_ngb'+suf] #, pre+'train_oneshot_c_w_ngb'+suf]
train_file_name_b = [pre+'train_oneshot_b_w_ngb'+suf, pre+'train_oneshot_c_w_ngb'+suf] #, pre+'train_oneshot_a_w_ngb'+suf]

# warm, warm_2 - used for warm-up training
train_file_name_warm = [pre+'test_oneshot_a'+suf]
train_file_name_warm_2 = [pre+'test_oneshot_b'+suf]

test_file_name = [pre+'test_test_w_ngb'+suf]

# the following are indices for features (excluding label)
# 0 - user_id,  1 - movie_id, 2 - gender, 3 - age, 4 - occ, 5 - release year, 6 - title (mul-hot), 7 - genres (mul-hot)
# tar_idx - whose emb to be generated
# attr_idx - which are intrinsic item attributes
tar_idx = [1]
# must be from small to large
attr_idx = [5,6,7]

n_ft = 11134
input_format = 'tfrecord' #'csv'
time_style = '%Y-%m-%d %H:%M:%S'
rnd_seed = 123 # random set (different seeds lead to different results)
att_dim = 10*len(attr_idx)
batch_size = 128 # used for warm up training

# meta_mode: self - use the new ad's own attributes
# ngb - use ngbs' pre-trained ID embs.
meta_mode = 'GME-A' # 'self', 'ngb', 'GME-P', 'GME-G', 'GME-A'
meta_batch_size_range = [60]
# learning rate for getting a new adapted embedding
cold_eta_range = [1e-4] # [0.05, 0.1]
# learning rate for meta learning
meta_eta_range = [5e-3] # [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
# learning rate for warm-up training
eta_range = [1e-3]
n_epoch = 1 # number of times to loop over the warm-up training data set
n_epoch_meta = 1 # number of times to loop over the meta training data set
alpha = 0.1
gamma = 1.0
test_batch_size = 128
# whether to perform warm up training
# only valid for 'gme_all_in_one_warm_up.py'
warm_up_bool = False # True
#################

save_model_ind = 0
# load emb and FC layer weights from a pre-trained DNN model
model_loading_addr = './tmp/dnn_1011_1705/'
output_file_name = '0801_0900'
k = 10 # embedding size / number of latent factors
opt_alg = 'Adam' # 'Adagrad'
kp_prob = 1.0
record_step_size = 200 # record the loss and auc after xx steps

