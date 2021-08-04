#!/bin/bash

name=big_train_main
cat ${name}.csv | python tfrecord_writer_new.py ${name}
name=train_oneshot_a_w_ngb
cat ${name}.csv | python tfrecord_writer_new.py ${name}
name=train_oneshot_b_w_ngb
cat ${name}.csv | python tfrecord_writer_new.py ${name}
name=train_oneshot_c_w_ngb
cat ${name}.csv | python tfrecord_writer_new.py ${name}
name=train_oneshot_d_w_ngb
cat ${name}.csv | python tfrecord_writer_new.py ${name}
name=test_oneshot_a
cat ${name}.csv | python tfrecord_writer_new.py ${name}
name=test_oneshot_b
cat ${name}.csv | python tfrecord_writer_new.py ${name}
name=test_oneshot_c
cat ${name}.csv | python tfrecord_writer_new.py ${name}
name=test_test
cat ${name}.csv | python tfrecord_writer_new.py ${name}
name=test_test_w_ngb
cat ${name}.csv | python tfrecord_writer_new.py ${name}

