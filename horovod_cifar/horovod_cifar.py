import horovod.tensorflow as hvd
import tensorflow as tf

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
# cpus = tf.config.experimental.list_physical_devices('CPU')
# # for cpu in cpus:
# #     tf.config.experimental.set_memory_growth(gpu, True)
# if cpus:
#     tf.config.experimental.set_visible_devices(cpus[hvd.local_rank()], 'CPU')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# (train_images, train_labels), _ = \
#     tf.keras.datasets.cifar10.load_data(path='cifar-%d.npz' % hvd.rank())
(train_images, train_labels), _ = \
    tf.keras.datasets.cifar10.load_data()
# dataset = tf.data.Dataset.from_tensor_slices(
#     (tf.cast(train_images / 255.0, tf.float32),
#              tf.cast(train_labels, tf.int64))
# )
# dataset = dataset.repeat().shuffle(10000).batch(128)
train_images = train_images / 255.0
dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)).shuffle(10000).repeat().batch(128)
cifar_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
print("model initialized")
loss = tf.losses.SparseCategoricalCrossentropy()

# Horovod: adjust learning rate based on number of GPUs.
opt = tf.optimizers.Adam(0.001 * hvd.size())

checkpoint_dir = './checkpoints'
checkpoint = tf.train.Checkpoint(model=cifar_model, optimizer=opt)
print('checkpoints created')


@tf.function
def training_step(images, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = cifar_model(images, training=True)
        loss_value = loss(labels, probs)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape)

    grads = tape.gradient(loss_value, cifar_model.trainable_variables)
    opt.apply_gradients(zip(grads, cifar_model.trainable_variables))

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        hvd.broadcast_variables(cifar_model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    return loss_value


# Horovod: adjust number of steps based on number of GPUs.
for batch, (images, labels) in enumerate(dataset.take(10000)):
    loss_value = training_step(images, labels, batch == 0)

    if batch % 10 == 0 and hvd.local_rank() == 0:
        print('Step #%d\tLoss: %.6f' % (batch, loss_value))

# Horovod: save checkpoints only on worker 0 to prevent other workers from
# corrupting it.
if hvd.rank() == 0:
    checkpoint.save(checkpoint_dir)
