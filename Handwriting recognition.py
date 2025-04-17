import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K 
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# GPU acceleration setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.optimizer.set_jit(True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("GPU acceleration enabled.")
    except RuntimeError as e:
        print(e)

# Load datasets
train = pd.read_csv('C:/Users/sures/Desktop/neuralnetwork/datasets/train_labels.csv')
valid = pd.read_csv('C:/Users/sures/Desktop/neuralnetwork/datasets/written_name_validation_v2.csv')

# Show sample training images
plt.figure(figsize=(15, 10))
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir =  'C:/Users/sures/Desktop/neuralnetwork/datasets/train/' + train.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        plt.imshow(image, cmap='gray')
        plt.title(train.loc[i, 'IDENTITY'], fontsize=12)
        plt.axis('off')
plt.subplots_adjust(wspace=0.2, hspace=-0.8)
plt.show()

# Clean the data
print("Number of NaNs in train set      : ", train['IDENTITY'].isnull().sum())
print("Number of NaNs in validation set : ", valid['IDENTITY'].isnull().sum())
train.dropna(inplace=True)
valid.dropna(inplace=True)

# Filter out 'UNREADABLE'
train = train[train['IDENTITY'] != 'UNREADABLE']
valid = valid[valid['IDENTITY'] != 'UNREADABLE']
train['IDENTITY'] = train['IDENTITY'].str.upper()
valid['IDENTITY'] = valid['IDENTITY'].str.upper()
train.reset_index(inplace=True, drop=True)
valid.reset_index(inplace=True, drop=True)

# Preprocessing
def preprocess(img):
    (h, w) = img.shape
    final_img = np.ones([64, 256]) * 255
    if w > 256:
        img = img[:, :256]
    if h > 64:
        img = img[:64, :]
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

# Prepare training and validation images
train_size = 30000
valid_size = 3000
train_x = []
valid_x = []

for i in range(train_size):
    img_dir = 'C:/Users/sures/Desktop/neuralnetwork/datasets/train/' + train.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image = preprocess(image)
        image = image / 255.
        train_x.append(image)

for i in range(valid_size):
    img_dir = 'C:/Users/sures/Desktop/neuralnetwork/datasets/validation/' + valid.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image = preprocess(image)
        image = image / 255.
        valid_x.append(image)

train_x = np.array(train_x).reshape(-1, 256, 64, 1)
valid_x = np.array(valid_x).reshape(-1, 256, 64, 1)

np.save('train_x.npy',train_x)
np.save('valid_x.npy',valid_x)

# Label encoding
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24
num_of_characters = len(alphabets) + 1
num_of_timestamps = 64

def label_to_num(label):
    return np.array([alphabets.find(ch) for ch in label])

def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:
            break   
        else:
            ret += alphabets[ch]
    return ret

# Prepare labels
train_y = np.ones([train_size, max_str_len]) * -1
train_label_len = np.zeros([train_size, 1])
train_input_len = np.ones([train_size, 1]) * (num_of_timestamps - 2)
train_output = np.zeros([train_size])

for i in range(train_size):
    label = label_to_num(train.loc[i, 'IDENTITY'])
    train_y[i, 0:len(label)] = label
    train_label_len[i] = len(label)

valid_y = np.ones([valid_size, max_str_len]) * -1
valid_label_len = np.zeros([valid_size, 1])
valid_input_len = np.ones([valid_size, 1]) * (num_of_timestamps - 2)
valid_output = np.zeros([valid_size])

for i in range(valid_size):
    label = label_to_num(valid.loc[i, 'IDENTITY'])
    valid_y[i, 0:len(label)] = label
    valid_label_len[i] = len(label)

# Model
input_data = Input(shape=(256, 64, 1), name='input')
inner = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(input_data)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2))(inner)

inner = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(2, 2))(inner)
inner = Dropout(0.3)(inner)

inner = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(inner)
inner = BatchNormalization()(inner)
inner = Activation('relu')(inner)
inner = MaxPooling2D(pool_size=(1, 2))(inner)
inner = Dropout(0.3)(inner)

inner = Reshape(target_shape=(64, 1024))(inner)
inner = Dense(64, activation='relu', kernel_initializer='he_normal')(inner)

inner = Bidirectional(LSTM(256, return_sequences=True))(inner)
inner = Bidirectional(LSTM(256, return_sequences=True))(inner)

inner = Dense(num_of_characters, kernel_initializer='he_normal')(inner)
y_pred = Activation('softmax', name='softmax')(inner)

model = Model(inputs=input_data, outputs=y_pred)
model.summary()

# CTC loss
labels = Input(name='gtruth_labels', shape=[max_str_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
model_final = Model(inputs=[input_data, labels, input_length, label_length], outputs=ctc_loss)
model_final.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(learning_rate=0.0001))

# Train
checkpoint=ModelCheckpoint('model_checkpoint.h5',save_best_only=True,monitor='val_loss',mode='min')
model_final.fit(
    x=[train_x, train_y, train_input_len, train_label_len], y=train_output,
    validation_data=([valid_x, valid_y, valid_input_len, valid_label_len], valid_output),
    epochs=60, batch_size=128
)
model_final.save('handwriting_model.h5')
# Evaluate
preds = model.predict(valid_x)
decoded = K.get_value(K.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1], greedy=True)[0][0])
prediction = [num_to_label(seq) for seq in decoded]

y_true = valid['IDENTITY'].values[:valid_size]

correct_char, total_char, correct = 0, 0, 0

for pr, tr in zip(prediction, y_true):
    total_char += len(tr)
    correct_char += sum([1 for x, y in zip(pr, tr) if x == y])
    if pr == tr:
        correct += 1

print('Correct characters predicted : %.2f%%' % (correct_char * 100 / total_char))
print('Correct words predicted      : %.2f%%' % (correct * 100 / valid_size))

predictions_df = pd.DataFrame({
    'True Label':y_true,
    'Predicted Label':prediction
})
predictions_df.to_csv('prediction.csv',index=False)

# Test predictions
test = pd.read_csv('C:/Users/sures/Desktop/neuralnetwork/datasets/test_labels.csv')
plt.figure(figsize=(15, 10))
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    img_dir = 'C:/Users/sures/Desktop/neuralnetwork/datasets/test/' + test.loc[i, 'FILENAME']
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Could not load image: {img_dir}")
        continue

    plt.imshow(image, cmap='gray')
    image = preprocess(image)
    image = image / 255.
    pred = model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], greedy=True)[0][0])
    plt.title(num_to_label(decoded[0]), fontsize=12)
    plt.axis('off')
plt.savefig('test_prediction.png')
plt.subplots_adjust(wspace=0.2, hspace=-0.8)
plt.show()