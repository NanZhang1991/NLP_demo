import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds



dataset = tf.data.experimental.make_csv_dataset("D:/source/imdb/IMDB_Dataset.csv", 
    batch_size=2, shuffle_buffer_size=1000, shuffle_seed=5)
iterator = dataset.as_numpy_iterator()
print(dict(next(iterator)))


(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    path="D:/source/imdb/imdb.npz", num_words=10000)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=80)
print(x_train.shape)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=80)


batchsz = 40
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz)
db_val = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_val = db_val.shuffle(1000).batch(batchsz)


model = keras.Sequential([keras.layers.Embedding(10000, 100, input_length=80),
                          keras.layers.GlobalAveragePooling1D(),
                          keras.layers.Dense(100),
                          keras.layers.Dense(1)])
model.summary()


model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(db_train, epochs=2,
                    validation_data=db_val, validation_steps=20)

output = model.predict(db_train)
print(output.shape)


