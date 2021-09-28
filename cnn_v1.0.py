import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
from PIL import Image
import os.path
from scipy.stats import gaussian_kde
import seaborn as sns
import time 

INPUT_SHAPE = 512
BATCH_SIZE = 32
EPOCHS = 5
PATH_TO_DATA_DIR = "../"
def load_data():
    train_df = pd.read_csv(PATH_TO_DATA_DIR + '4771-sp20-covid/train.csv')
    test_df = pd.read_csv(PATH_TO_DATA_DIR + '4771-sp20-covid/test.csv')

    train_id = train_df['id'].values
    train_filename = train_df['filename'].values
    train_label = train_df['label'].values

    test_id = test_df['id'].values
    test_filename = test_df['filename'].values

    return (train_id, train_filename, train_label), (test_id, test_filename)

# [covid, viral, bacterial, normal]
def create_lbls_as_vectors(lbls):
    dim = len(lbls)
    y = [np.zeros(4) for i in range(dim)]
    for i, lbl in enumerate(lbls):
        if lbl == 'covid':
            y[i][0] = 1
        elif lbl == 'viral':
            y[i][1] = 1
        elif lbl == 'bacterial':
            y[i][2] = 1
        else:
            y[i][3] = 1

    return y

def create_lbls(lbls=''):
    if os.path.isfile('training_labels.npy'):
        print('Found training_labels.npy file! Loaded from that.')
        return np.load('training_labels.npy', allow_pickle=False)

    dim = len(lbls)
    y = np.zeros(dim)
    for i, lbl in enumerate(lbls):
        if lbl == 'covid':
            y[i] = 0
        elif lbl == 'viral':
            y[i] = 1
        elif lbl == 'bacterial':
            y[i] = 2
        else:
            y[i] = 3

    np.save('training_labels.npy', y)
    return y

def create_ims(img_files='', t='train'):
    if t == 'train':
        filet = 'training_images_%d.npy' % (INPUT_SHAPE)
    if t == 'test':
        filet = 'testing_images_%d.npy' % (INPUT_SHAPE)

    if os.path.isfile(filet):
        print('Found %s file! Loaded from that.' % (filet))
        return np.load(filet, allow_pickle=False)

    wd = PATH_TO_DATA_DIR + '4771-sp20-covid/'
    ds = []
    for f in img_files:
        im = Image.open(wd + f)
        im = im.resize((INPUT_SHAPE, INPUT_SHAPE))
        im = np.array(im)
        im = im / 255
        ds.append(im)

    np.save(filet, np.array(ds))
    return np.array(ds)

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(INPUT_SHAPE, INPUT_SHAPE, 1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def write_preds_file(file_name, preds):
    f = open(file_name, "w")
    f.write("Id,label\n")
    for i, pred in enumerate(preds):
        cid = i
        cpred = np.argmax(pred)
        if cpred == 0:
            fpred = "covid"
        elif cpred == 1:
            fpred = "viral"
        elif cpred == 2:
            fpred = "bacterial"
        else:
            fpred = "normal"
        f.write("%s,%s\n" % (cid, fpred))
    f.close()

if __name__ == '__main__':
    desired_file1 = 'training_images_%d.npy' % (INPUT_SHAPE)
    desired_file2 = 'testing_images_%d.npy' % (INPUT_SHAPE)
    if os.path.isfile('training_labels.npy') and os.path.isfile(desired_file1) and os.path.isfile(desired_file2):
        # skip loading step
        train_y = create_lbls()
        train_im = create_ims()
        test_im = create_ims(t='test')
    else:
        # do load for everything
        (train_id, train_filename, train_label), (test_id, test_filename) = load_data()
        train_y = create_lbls(train_label)
        train_im = create_ims(img_files=train_filename)
        test_im = create_ims(img_files=test_filename, t='test')

    train_im = train_im.reshape(1127, INPUT_SHAPE, INPUT_SHAPE, 1)
    test_im = test_im.reshape(484, INPUT_SHAPE, INPUT_SHAPE, 1)

    model = create_model()
    mdl_file = 'cnn_v1_%d' % (INPUT_SHAPE)
    if os.path.isfile(mdl_file + '.input'):
        model.load_weights(mdl_file)
        print("Loaded model from saved file!")
    else:
        print("Training starting now!\n")
        start = time.process_time()
        model.fit(train_im, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS)
        end = time.process_time()
        print("Training done! Total training time: %.2gs" % (end-start))
        model.save_weights(mdl_file)
    
    model.summary()

    pds_file = 'preds_%d.npy' % (INPUT_SHAPE)
    if os.path.isfile(pds_file):
        predictions = np.load(pds_file)
    else:
        predictions = model.predict(test_im)
        np.save(pds_file, predictions)

    write_preds_file(file_name='submission_v1.0.0_%d.csv' % (INPUT_SHAPE), preds=predictions)
