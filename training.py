import os
import numpy as np 
import pandas as pd 
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.utils import class_weight
from tensorflow.keras.metrics import AUC

from efficientnet.tfkeras import EfficientNetB3
from efficientnet.tfkeras import EfficientNetB2
from tensorflow.python.keras.applications import InceptionV3


auc = AUC(name='auc')

#%%

print("EfficientNet")

root = './input/vehicle/train/train/'
data = []

for category in sorted(os.listdir(root)):
    for file in sorted(os.listdir(os.path.join(root, category))):
        data.append((category, os.path.join(root, category,  file)))

df = pd.DataFrame(data, columns=['class', 'file_path'])

skf = StratifiedKFold(n_splits = 10)

X = df['file_path']
y = df['class']

verkot = []

kokonaisAika = time.time()
parhaat_acc = []
parhaat_auc = []

batch_size = 11

#paikka = 0

#%% K-folds to files so that they can be loaded for different networks

i = 0
for train_index, test_index in skf.split(X,y):
    
    trainNimi= "data_train_foldi{}".format(i)
    valNimi = "data_val_foldi{}".format(i)
        
    data_train = df.iloc[train_index]
    data_test = df.iloc[test_index]
    
    data_train.to_pickle(trainNimi)
    data_test.to_pickle(valNimi)
        
    i += 1


#%% Change parameters as needed, trying to keep things simple

for i in range(0,2):

    K.clear_session()    
    
    print("##################################")
    print("FOLDI {:1d}".format(i))
    print("##################################")
          
    trainNimi= "data_train_foldi{}".format(i)
    valNimi = "data_val_foldi{}".format(i)
    
    print("Ladataan: {}".format(trainNimi))    
    data_train = pd.read_pickle(trainNimi)
    
    print("Ladataan: {}".format(valNimi))
    data_test = pd.read_pickle(valNimi)
             
    opetusAika = time.time()
           
    valdatagen = ImageDataGenerator(rescale=1./255)
    
    traindatagen = ImageDataGenerator(
            rescale = 1./255,
            horizontal_flip = True,
            rotation_range = 25,
            height_shift_range = 0.2,
            width_shift_range = 0.2,
            zoom_range = 0.2,
            shear_range = 0.2,
            brightness_range = (0.9, 1.1)
            )
            
    train_generator = traindatagen.flow_from_dataframe(
            dataframe = data_train,
            x_col = 'file_path',
            y_col = 'class',
            target_size = (300,300),
            batch_size = batch_size            
            )
    
    val_generator = valdatagen.flow_from_dataframe(
            dataframe = data_test,
            x_col = 'file_path',
            y_col = 'class',
            target_size = (300,300),
            batch_size = batch_size            
            )
    
    esiverkko = EfficientNetB3(include_top = False, weights = 'imagenet', input_shape = (300,300,3))    
    
    verkko = models.Sequential()
    verkko.add(esiverkko)
    verkko.add(layers.GlobalMaxPooling2D())
    verkko.add(layers.Dropout(rate = 0.2))
#    verkko.add(layers.Dense(1000, activation = 'relu', kernel_initializer = 'he_uniform'))   # for InceptionV3
#    verkko.add(layers.Dropout(rate = 0.70)) # for InceptionV3
#    verkko.add(layers.BatchNormalization()) # for InceptionV3       
    verkko.add(layers.Dense(17, activation = 'softmax',kernel_initializer = 'he_uniform'))
        
    verkko.compile(loss = 'categorical_crossentropy',
                   optimizer = optimizers.RMSprop(lr=1e-5),
                   metrics = ['acc', auc])
    
    verkko.summary()
    
    train_steps = int(np.ceil( len(data_train) / batch_size))
    val_steps = int(np.ceil(len(data_test) / batch_size))
    
    tallennuspaikka = "paras_acc.hdf5" 
    auctallennus = "paras_auc.hdf5"
            
    tallennus = ModelCheckpoint( monitor = 'val_acc',  mode = 'max', filepath = tallennuspaikka, verbose = 1, save_weights_only = True, save_best_only=True)           
    tallennus_auc = ModelCheckpoint( monitor = 'val_auc',  mode = 'max', filepath = auctallennus, verbose = 1, save_weights_only = True, save_best_only=True)           
    
    vahentaja =  ReduceLROnPlateau( monitor='val_acc', factor=0.5, patience=3, verbose=1)
    
    stoppi = EarlyStopping(monitor = 'val_acc', patience = 10, verbose = 1)
    
    # Easy way to fight the class imbalance using sklearn 
    luokkapainot = class_weight.compute_class_weight('balanced', np.unique(y), y)
      
    historia = verkko.fit_generator( train_generator,
                            steps_per_epoch = train_steps,
                            epochs = 100,
                            validation_data = val_generator,
                            validation_steps = val_steps,
                            verbose = 1,
                            callbacks = [tallennus, vahentaja, tallennus_auc, stoppi],
                            workers = 6,
                            max_queue_size = 64,
                            class_weight = luokkapainot
                          )
    
    paras_auc = max(historia.history['val_auc'])
    parhaat_auc.append(paras_auc)
    aucnimi = "01_EffiNetB3_auc_foldi_{}.hdf5".format(i)
    print("Tallennetaan: {}".format(aucnimi))
    verkko.load_weights(auctallennus)
    verkko.save(aucnimi)
    
    verkko.load_weights(tallennuspaikka)
    paras = max(historia.history['val_acc'])
    parhaat_acc.append(paras)
       
    nimi = "01_EffiNetB3_acc_foldi_{}.hdf5".format(i)
    print("Tallennetaan: {}".format(nimi))
    verkko.save(nimi)
    
    plt.figure(figsize=(30,20))
    plt.plot(historia.history['acc'])
    plt.plot(historia.history['val_acc'])
    plt.yscale('logit')
    plt.title("Malli ACC densella: {}".format(1000))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'Val_acc'], loc='upper left')
    plt.show()
        
    plt.figure(figsize=(30,20))
    plt.plot(historia.history['auc'])
    plt.plot(historia.history['val_auc'])
    plt.yscale('logit')
    plt.title("Malli AUC densella: {}".format(1000))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend([ 'Train_auc', 'Val_auc'], loc='upper left')
    plt.show()

    K.clear_session()    
    
    print("Aikaa meni {:5.3f} h".format( (time.time() - opetusAika)/(60**2) ))