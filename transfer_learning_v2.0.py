import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
from PIL import Image
import os.path
from scipy.stats import gaussian_kde
import seaborn as sns
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow_hub as hub
import time
from sklearn.metrics import confusion_matrix

INPUT_SHAPE = 224
IMAGE_DIM = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40
PATH_TO_DATA_DIR = "../"
CHECKPOINT_PATH = "checkpoints2/cp-{epoch:04d}.ckpt"
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
METRICS = [keras.metrics.Accuracy(name='accuracy'), 
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall')]

def load_data():
    train_df = pd.read_csv(PATH_TO_DATA_DIR + '4771-sp20-covid/train.csv')
    test_df = pd.read_csv(PATH_TO_DATA_DIR + '4771-sp20-covid/test.csv')

    train_id = train_df['id'].values
    train_filename = train_df['filename'].values
    train_label = train_df['label'].values

    test_id = test_df['id'].values
    test_filename = test_df['filename'].values

    """
    list_ds = tf.data.Dataset.list_files('4771-sp20-covid/train/*')
    for f in list_ds.take(5):
          print(f.numpy())
    """
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
        im = im / 255.0
        im = np.stack([im,im,im], axis=2)
        ds.append(im)

    np.save(filet, np.array(ds))
    return np.array(ds)

def create_model():
    #classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
    #model = tf.keras.Sequential([hub.KerasLayer(classifier_url, input_shape=IMAGE_DIM+(3,))])
    feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,input_shape=(224,224,3))
    feature_extractor_layer.trainable = False
    model = tf.keras.models.Sequential([feature_extractor_layer,
                                        tf.keras.layers.Dense(4, activation='softmax')])

    '''
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(INPUT_SHAPE, INPUT_SHAPE, 1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')])
    '''
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


def error_analysis(y_hat, y):
    print('y', len(y))
    print('yhat', len(y_hat))
    correct = np.zeros(4)
    for i in range(len(y)):
        if np.argmax(y_hat[i]) == y[i]:
            correct[int(y[i])] += 1
    
    wtotal = (5.0/77.0 * correct[0]) + (correct[1] / 350.0) + (correct[2] / 350) + (correct[3] / 350)
    wacc = wtotal / 8.0
    print('Weighted metric score: ', wacc)
    print('Breakdown [cov, vir, bac, nor]: ', correct)

def plot_metrics(history):
    metrics =  ['loss', 'accuracy']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(1,2,n+1)
        plt.plot(history.epoch,  history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()
    plt.show()


def interpret(model, train_im, train_y):
    originals = [1031, 916, 1104, 1102]
    f, axarr = plt.subplots(len(originals),3)
    CONVOLUTION_NUMBER = 1
    key = ['Covid', 'Viral', 'Bacterial', 'Healthy']
    ims = np.array([train_im[originals[t]] for t in range(len(originals))]).reshape(len(originals), INPUT_SHAPE, INPUT_SHAPE, 3)
    preds = model.predict(ims)
    fpreds = [key[int(np.argmax(preds[p]))] for p in range(len(preds))]
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
    for i in range(len(originals)):
        axarr[i,0].imshow(ims[i].reshape(INPUT_SHAPE, INPUT_SHAPE, 3), cmap=plt.cm.binary)
        axarr[i,0].grid(False)
        axarr[i,0].set_xlabel("%s x-ray" % (key[int(train_y[originals[i]])]))
        axarr[i,0].set_ylabel("Image %d/1127" % (originals[i]))
        axarr[i,0].get_xaxis().set_ticks([])
        axarr[i,0].get_yaxis().set_ticks([])
        axarr[i,2].text(0.5, 0.5, "Predicted:\n" + fpreds[i], horizontalalignment='center', verticalalignment='center')
        axarr[i,2].grid(False)
        axarr[i,2].get_xaxis().set_ticks([])
        axarr[i,2].get_yaxis().set_ticks([])

    for x in range(0,1):
        for y in range(len(originals)):
            f1 = activation_model.predict(ims[y].reshape(1, INPUT_SHAPE, INPUT_SHAPE, 3))[x] # could change 3 back to 1
            print(type(f1))
            print(f1)
            print(f1.shape)
            #axarr[y,x+1].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
            axarr[y,x+1].imshow(np.reshape(f1, (40,32)), cmap='inferno')
            axarr[y,x+1].grid(False)
            axarr[y,x+1].get_xaxis().set_ticks([])
            axarr[y,x+1].get_yaxis().set_ticks([])
            if x % 2 == 0:
                if x == 0:
                    axarr[y,x+1].set_xlabel("Convolutional Layer 1")
                else:
                    axarr[y,x+1].set_xlabel("Convolutional Layer 2")
            else:
                if x == 1:
                    axarr[y,x+1].set_xlabel("Max Pooling Layer 1")
                else:
                    axarr[y,x+1].set_xlabel("Max Pooling Layer 2")
    plt.show()

def confusion():
    train_df = pd.read_csv(PATH_TO_DATA_DIR + '4771-sp20-covid/train.csv')
    preds_df = pd.read_csv('training.csv')
    correct_labels = train_df['label'][-226:].tolist()
    correct_ids = train_df['id'][-226:].tolist()
    predicted_labels = preds_df['label'].tolist()
    print(type(correct_ids))
    tabs = [[[], []] for x in range(4)]
    key = ['covid', 'viral', 'bacterial', 'normal']
    for n in range(len(predicted_labels)):
        clab = correct_labels[n]
        cidx = correct_ids[n]
        cpred = predicted_labels[n]
        tabs[key.index(clab)][int(clab==predicted_labels[n])].append((cidx, cpred))
    
    for n in range(4):
        print(key[n])
        print("Num correctly predicted: ", len(tabs[n][1]))
        print("Correctly predicted: ", tabs[n][1])
        print("\n")
        print("Num incorrectly predicted: ", len(tabs[n][0]))
        print("Incorrectly predicted: ", tabs[n][0])
        print("\n\n")

    mtx = confusion_matrix(correct_labels, predicted_labels, labels=["covid", "viral", "bacterial", "normal"])
    df_cm = pd.DataFrame(mtx, ['covid', 'viral', 'bacterial', 'normal'], ['covid', 'viral', 'bacterial', 'normal'])
    ax = plt.axes()
    sns.heatmap(df_cm, annot=True, ax=ax, cbar=False)
    ax.set_xlabel("Predicted Class", labelpad=15, fontsize=14)
    ax.set_ylabel("Actual Class", labelpad=15, fontsize=14)
    plt.yticks(rotation=0)
    ax.set_title("Transfer Learning v.2.0 Confusion Matrix of the Validation Data (after 20 epochs of training)", fontsize=18, pad=20)
    plt.show()

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

    train_im = train_im.reshape(1127, INPUT_SHAPE, INPUT_SHAPE, 3)
    test_im = test_im.reshape(484, INPUT_SHAPE, INPUT_SHAPE, 3)

    model = create_model()
    #if os.path.isfile(pds_file):
    '''
    new_preds = model.predict(train_im[:500])
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    pred_cn = imagenet_labels[np.argmax(new_preds, axis=-1)]
    #print(pred_cn)
    d = dict()
    dref = dict()
    for i, cls in enumerate(train_y[:500]):
        if cls in d:
            # append the new number to the existing array at this slot
            #if pred_cn[i] not in d[cls]:
            d[cls].append(pred_cn[i])
            if pred_cn[i] not in dref[cls]:
                dref[cls].append(pred_cn[i])
        else:
            # create a new array in this slot
            d[cls] = [pred_cn[i]]
            dref[cls] = [pred_cn[i]]

    for cls in d.keys():
        print('Class: ', cls)
        for entry in dref[cls]:
            print('%s occurs %d times' % (entry, d[cls].count(entry)))
    '''
    """
    model.save_weights(CHECKPOINT_PATH.format(epoch=0))
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 verbose=0, 
                                                 save_weights_only=True,
                                                 period=4)
    class_weight = {0: 5., 1: 1., 2: 1., 3: 1.}
    #callbacks=[EarlyStopping(patience=3, restore_best_weights=True), ReduceLROnPlateau(patience=2)]
    history = model.fit(train_im, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, shuffle=True, class_weight=class_weight, callbacks=[cp_callback])
    train_predictions = model.predict(train_im)
    error_analysis(train_predictions, train_y)
    model.summary()
    plot_metrics(history)
    """
    #pds_file = 'preds_%d.npy' % (INPUT_SHAPE)
    #if os.path.isfile(pds_file):
        #predictions = np.load(pds_file)
    #else: # tab the following 2 lines when uncommenting this block
    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    print(latest)
    model.load_weights(latest)
    predictions = model.predict(train_im[-226:])
    write_preds_file(file_name='training.csv', preds=predictions)
    confusion()
    #interpret(model, train_im, train_y)
    #predictions = model.predict(test_im)
    #write_preds_file(file_name='submission_v2.0_%d.csv' % (INPUT_SHAPE), preds=predictions)
