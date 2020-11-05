import json
import os

import tensorflow as tf

import cifar10

per_worker_batch_size = 64
print(os.environ['TF_CONFIG'])
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = cifar10.cifar_dataset(global_batch_size)

with strategy.scope():
    multi_worker_model = cifar10.build_and_compile_cnn_model()

multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
