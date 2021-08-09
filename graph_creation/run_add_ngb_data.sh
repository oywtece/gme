ft_dict_fn=movie_id_ft_dict.txt

for i in a b c d
do
  input_fn=train_oneshot_$i
  ngb_dict_fn=${input_fn}_movie_id_ngb_dict.txt 
  cat ${input_fn}.csv | python add_ngb_data.py $ft_dict_fn $ngb_dict_fn > ${input_fn}_w_ngb.csv
done

for i in a b c
do
  input_fn=test_oneshot_$i
  ngb_dict_fn=${input_fn}_movie_id_ngb_dict.txt
  cat ${input_fn}.csv | python add_ngb_data.py $ft_dict_fn $ngb_dict_fn > ${input_fn}_w_ngb.csv
done

input_fn=test_test
ngb_dict_fn=${input_fn}_movie_id_ngb_dict.txt
cat ${input_fn}.csv | python add_ngb_data.py $ft_dict_fn $ngb_dict_fn > ${input_fn}_w_ngb.csv

