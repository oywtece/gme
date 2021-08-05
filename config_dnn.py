'''
config file
'''

n_one_hot_slot = 6
n_mul_hot_slot = 2
max_len_per_slot = 5 # max num of fts in each mul-hot slot
num_csv_col = 17 # num of cols in csv file
layer_dim = [256, 128, 1]

pre = './data/'
suf = '.tfrecord'
train_file_name = [pre+'big_train_main'+suf]
test_file_name = [pre+'test_oneshot_a'+suf]

n_ft = 11134
input_format = 'tfrecord' #'csv'
time_style = '%Y-%m-%d %H:%M:%S'
rnd_seed = 123 # random seed (different seeds lead to different results)

# if you want to tune model paras, do not save model; you can enter several values in 'eta_range' and 'batch_size_range'
# if you want to save model, then enter only one value in 'eta_range' and 'batch_size_range'
save_model_ind = 0 # 1 - save model, 0 - do not save model
eta_range = [1e-3] #  [1e-4, 5e-4, 1e-3]
batch_size_range = [128]

# if you save model, then the model will be saved at './tmp/dnn_' + output_file_name
output_file_name = '0801_0900'
k = 10 # embedding size / number of latent factors
opt_alg = 'Adam' # 'Adagrad' # 'Adam'
kp_prob = 1.0
n_epoch = 1 # number of times to loop over the whole data set
record_step_size = 2000 # record the loss and auc after xx steps

