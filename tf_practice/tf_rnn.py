import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging


logging.basicConfig()


dataset = tf.data.experimental.make_csv_dataset(
    "D:/source/imdb/IMDB_Dataset.csv",
    batch_size=2, shuffle_buffer_size=1000, shuffle_seed=5)
iterator = dataset.as_numpy_iterator()
logging.info(dict(next(iterator)))


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


word_index = keras.datasets.imdb.get_word_index(
    path="D:/source/imdb/imdb_word_index.json",)
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


logging.info(decode_review(x_train[0]))


total_words = 10000
embedding_len = 100
max_review_len = 80


class RNNFromCell(keras.Model):
    # 多层 SimpleRNNCell 网络
    # Cell 方式构建多层网络
    def __init__(self, units):
        super().__init__()
        # [b, 64]，构建 Cell 初始化状态向量，重复使用
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]
        # 词向量编码 [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)
        # 构建 2 个 Cell
        self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)
        # 构建分类网络，用于将 CELL 的输出特征进行分类，2 分类
        # [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs  # [b, 80]
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute,[b, 80, 100] => [b, 64]
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):  # word: [b, 100]
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
        # 末层最后一个输出作为分类网络的输入: [b, 64] => [b, 1]
        x = self.outlayer(out1)
        # p(y is pos|x)
        prob = tf.sigmoid(x)
        return prob


def trainer_rnn_cell():
    units = 128  # RNN 状态向量长度
    epochs = 2   # 训练 epochs
    model = RNNFromCell(units)
    model.build(input_shape=(None, units))
    print(model.summary())
    # 装配
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    # 训练和验证
    model.fit(db_train, epochs=epochs, validation_data=db_val)
    # 测试
    model.evaluate(db_val)


def rnn_simple():
    model = keras.Sequential()
    model.add(keras.layers.Embedding(total_words, embedding_len,
              input_length=max_review_len))
    model.add(keras.layers.GRU(256, return_sequences=True))
    model.add(keras.layers.SimpleRNN(128))
    model.add(keras.layers.Dense(1))
    model.summary()
    return model


def trainer_rnn_simple(model):
    model = model
    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(db_train, epochs=2,
              validation_data=db_val, validation_steps=20)
    model.save('./model_dirs/imdb_model/imdb_rnn.h5')
    model = keras.models.load_model("./model_dirs/imdb_model/imdb_rnn.h5")
    model = tf.keras.models.load_model("./model_dirs/tf_model_savedmodel")


class RNN(keras.Model):
    def __init__(self):
        super(RNN, self).__init__()
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)
        self.GRU = layers.GRU(256, return_sequences=True)
        self.SimpleRNN = layers.SimpleRNN(128)
        self.Dense = layers.Dense(1)

    def call(self, inputs):
        x = inputs
        x = self.embedding(x)
        out = self.GRU(x)
        out = self.SimpleRNN(out)
        outputs = self.Dense(out)
        return outputs


if __name__ == "__main__":
    # trainer_rnn_simple(model=RNN())
    trainer_rnn_cell()
