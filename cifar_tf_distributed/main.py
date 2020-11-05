import json
import os
import sys

import tensorflow as tf

import cifar10


def main(index):
    tf_config = {
        'cluster': {
            'worker': ['localhost:12345', 'localhost:23456', 'localhost:34567']
        },
        'task': {'type': 'worker', 'index': index}
    }
    os.environ['TF_CONFIG'] = json.dumps(tf_config)

    per_worker_batch_size = 8

    num_workers = len(tf_config['cluster']['worker'])

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = cifar10.cifar_dataset(global_batch_size)

    with strategy.scope():
        multi_worker_model = cifar10.build_and_compile_cnn_model()

    multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)


if __name__ == '__main__':
    index = sys.argv[1]
    main(index)
