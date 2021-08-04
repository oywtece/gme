# Graph Meta Embedding (GME)
Graph Meta Embedding (GME) models for generating initial ID embeddings for cold-start ads in CTR/CVR prediction.

If you use this code, please cite the following paper:
* **Learning Graph Meta Embeddings for Cold-Start Ads in Click-Through Rate Prediction. In SIGIR, ACM, 2021.**

arXiv: https://arxiv.org/abs/2105.08909

ACM DL: https://dl.acm.org/doi/10.1145/3404835.3462879

#### Bibtex
```
@inproceedings{ouyang2021learning,
  title={Learning Graph Meta Embeddings for Cold-Start Ads in Click-Through Rate Prediction},
  author={Ouyang, Wentao and Zhang, Xiuwu and Ren, Shukui and Li, Li and Zhang, Kun and Luo, Jinmei and Liu, Zhaojie and Du, Yanlong},
  booktitle={SIGIR},
  pages={1157--1166},
  year={2021},
  organization={ACM}
}
```

#### TensorFlow (TF) version
1.12.0

#### Abbreviation
ft - feature, slot == field

## Data Preparation
Data is in the "csv" or the "tfrecord" format.
* Each csv row contains the label and then the features (first one-hot features and then multi-hot features).

Assume there are N unique fts. Fts need to be indexed from 1 to N. Use 0 for missing values or for padding.

We categorize fts as i) **one-hot** or **univalent** (e.g., user id, city) and ii) **mul-hot** or **multivalent** (e.g., words in title).

We need to prepare **two datasets**: one for the main CTR prediction model (including pre-training and warm-up training) and the other for the GME model (i.e., embedding generation model).

### 1) Dataset for the main CTR prediction model
One row of the csv data looks like:
* \<label\>\<one-hot fts\>\<mul-hot fts\>

We need to specify the following parameters for data partitioning after the data are loaded:
* n_one_hot_slot: num of one-hot fts
* n_mul_hot_slot: num of mul-hot fts
* max_len_per_slot: max num of fts in each mul-hot slot
 
#### Example 1
1) original fts (ft_name:ft_value)
* label:0, user_id:a, movie_id:b, title:c/d/e 
2) csv fts
* 0, a, b, c, d, e, 0, 0

#### Explanation 1
csv format:\
\<label\>\<one-hot fts\>\<mul-hot fts\>

csv format settings:\
n_one_hot_slot = 2 # user_id & movie_id \
n_mul_hot_slot = 1 # title \
max_len_per_slot = 5

For the mul-hot ft slot "title", we have 3 fts, which are c, d and e. Terefore, we pad 2 zeros (because max_len_per_slot = 5).
If there are more than max_len_per_slot fts, we keep only the first max_len_per_slot.

#### Differences btw pre-training and warm-up training
* Pre-training: use labeled old ads; the aim is to obtain the embedding matrix and FC layer weights
* Warm-up training: use a small number of labeled new ads; the aim is to obtain updated embeddings for new ad IDs (but keep other params unchanged)

### 2) Dataset for the GME model (use meta learning)
One row of the csv data looks like:
* \<label\>\<one-hot fts\>\<mul-hot fts\>\<ngb1 one-hot fts\>\<ngb1 mul-hot fts\>\<ngb2 one-hot fts\>\<ngb2 mul-hot fts\> ...

#### Example 2
1) original fts (ft_name:ft_value)
* label:0, user_id:a, movie_id:b, title:c/d/e, ngb_1_user_id:i, ngb_1_movie_id:j, ngb_1_title:k/l, ngb_2_user_id:s, ngb_2_movie_id:t, ngb_2_title:x/y/z/c/d/e
2) csv fts
* 0, a, b, c, d, e, 0, 0, i, j, k, l, 0, 0, 0, s, t, x, y, z, c, d

#### Explanation 2
csv format:\
\<label\>\<one-hot fts\>\<mul-hot fts\>\<ngb1 one-hot fts\>\<ngb1 mul-hot fts\>\<ngb2 one-hot fts\>\<ngb2 mul-hot fts\> ...

csv format settings:\
n_one_hot_slot = 2 # user_id & movie_id \
n_mul_hot_slot = 1 # title \
max_len_per_slot = 5

n_one_hot_slot_ngb = 2 # user_id & movie_id \
n_mul_hot_slot_ngb = 1 # title \
max_len_per_slot_ngb = 5

## Sample Data
In the **data** folder.\
Reformatted [MovieLens-1M](https://grouplens.org/datasets/movielens/) data (csv/tfrecord format with ft index). \
The scripts run **much faster** with the **tfrecord** data files.
We provide **tfrecord_writer_new.py** which can easily convert csv files to tfrecord files.

* We use 8 features: user_id, movie_id, gender, age, occ, release year, title (multi-hot), genres (multi-hot)

* The following csv files contains 1 + 16 cols (1 label + 6 one-hot slot + 2 multi-hot slots * 5 values each) \
big_train_main.csv \
test_oneshot_a.csv \
test_oneshot_b.csv \
test_oneshot_c.csv \
test_test.csv

* The following csv files contains 1 + 16 + 160 cols, with 10 neighbors appended (acc. to the target movie_id) \
train_oneshot_a_w_ngb.csv \
train_oneshot_b_w_ngb.csv \
train_oneshot_c_w_ngb.csv \
train_oneshot_d_w_ngb.csv \
test_test_w_ngb.csv

## Get Sample Data Ready
Go to the **data** folder.
```bash
unzip csv.zip
bash run_tfrecord_writer.sh
```

## Config
### Validation and hyperparameter tuning
You can set multiple values for hyperparameters. \
Example: eta_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

### Testing
Set only 1 value for the optimal hyperparameter found in the validation step. \
Example: eta_range = [0.02]

## Source Code
* config_dnn.py -- config file for DNN (main CTR prediction model)
* config_gme.py -- config file for GME models (cold-start embedding generation model)
* ctr_funcs.py -- functions
* dnn.py -- DNN model
* gme_all_in_one.py -- GME model, no warm-up training (meta_mode can be set to: 'self', 'ngb', 'GME-P', 'GME-G', 'GME-A')
* gme_all_in_one_warm_up.py - GME model, with warm-up training (need to set warm_up_bool=True)

## Run the Code
Train GME
```bash
nohup python gme_all_in_one.py > gme_[output_file_name].out 2>&1 &
```
Train DNN
* This step is **not necessary** because one saved DNN model is already provided in the 'tmp' folder.
```bash
nohup python dnn.py > dnn_[output_file_name].out 2>&1 &
```
