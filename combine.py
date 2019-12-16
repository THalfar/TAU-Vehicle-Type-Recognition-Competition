import os
import numpy as np 
import pandas as pd 
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time
from tensorflow.keras.models import load_model
from sklearn import preprocessing

import GeneticSVMrbf

print("Yhdistelyt")

kokoaika = time.time()

root = './input/vehicle/train/train/'
data = []

for category in sorted(os.listdir(root)):
    for file in sorted(os.listdir(os.path.join(root, category))):
        data.append((category, os.path.join(root, category,  file)))

df = pd.DataFrame(data, columns=['class', 'file_path'])
X = df['file_path']
y = df['class']

le = preprocessing.LabelEncoder()    
le.fit(y)
y = le.transform(y)

test_datagen = ImageDataGenerator(rescale = 1./255)

#%% InceptionV3

testi_askel = 32
train_generator = test_datagen.flow_from_dataframe(
        dataframe = df,
        x_col = 'file_path',
        y_col = None,
        batch_size = testi_askel,
        shuffle = False,
        class_mode = None,
        target_size = (299,299)
        )

steps = int(np.ceil(len(df) / testi_askel))

nimi = "05_IncV3_auc_foldi_4.hdf5"
print("Ladataan verkko {}".format(nimi))
verkko = load_model(nimi, compile = False)
verkko.summary()
 
tulokset = np.array(verkko.predict_generator(train_generator,
                                steps = steps,
                                verbose = 1,
                                workers = 7,
                                max_queue_size = 64))

K.clear_session()

for i in range(5,6):
        
    nimi = "05_IncV3_acc_foldi_{}.hdf5".format(i)
    print("Ladataan verkko: {}".format(nimi))
    verkko = load_model(nimi, compile = False)
    verkko.summary()

    train_generator.reset()
    
    pred = np.array(verkko.predict_generator(train_generator,
                                    steps = steps,
                                    verbose = 1,
                                    workers = 7,
                                    max_queue_size = 64))
    tulokset = np.hstack((tulokset, pred))
    
    K.clear_session()    

#%% EfficientNetB2

testi_askel = 22
train_generator = test_datagen.flow_from_dataframe(
        dataframe = df,
        x_col = 'file_path',
        y_col = None,
        batch_size = testi_askel,
        shuffle = False,
        class_mode = None,
        target_size = (260,260)
        )

steps = int(np.ceil(len(df) / testi_askel))

for i in range(2,4):
    
    train_generator.reset()
    
    nimi = "01_EffiNetB2_acc_foldi_{}.hdf5".format(i)
    print("Ladataan verkko {}".format(nimi))
    verkko = load_model(nimi, compile = False)
    verkko.summary()
         
    pred = np.array(verkko.predict_generator(train_generator,
                                    steps = steps,
                                    verbose = 1,
                                    workers = 7,
                                    max_queue_size = 64))
    
    
    tulokset = np.hstack((tulokset, pred))
    
    K.clear_session()
    
#%% EfficientNetB3
    
testi_askel = 11
train_generator = test_datagen.flow_from_dataframe(
    dataframe = df,
    x_col = 'file_path',
    y_col = None,
    batch_size = testi_askel,
    shuffle = False,
    class_mode = None,
    target_size = (300,300)
    )

steps = int(np.ceil(len(df) / testi_askel))


for i in range(0,2):
    
    train_generator.reset()
    
    nimi = "01_EffiNetB3_acc_foldi_{}.hdf5".format(i)
    print("Ladataan verkko {}".format(nimi))
    verkko = load_model(nimi, compile = False)
    verkko.summary()
        
    pred = np.array(verkko.predict_generator(train_generator,
                                    steps = steps,
                                    verbose = 1,
                                    workers = 7,
                                    max_queue_size = 64))
    
    
    tulokset = np.hstack((tulokset, pred))
    
    K.clear_session()
   

#%% SVC genetic optimization    
    
yhdistelyAika = time.time()
      
yhdistely = GeneticSVMrbf.GeneticSVMrbf(n_gen=10, size= 30, n_best=14, n_rand=1, 
                              n_children=2, mutation_rate=0.15, verbose = 2,
                              tulostetaanAikakausi = True, tulostetaanKehitys = False, 
                              aloituskerroin = 5, cv = 5, eliittikerroin = [6,4],
                              luokkapainot = True
                              )

yhdistely.fit(tulokset, y, 1)

yhdistelija = yhdistely.tuoEstimaattori()        
yhdistelija.fit(tulokset, y)

print("Aikaa meni yhdistelyn opetteluun {:5.3f} h".format( (time.time() - yhdistelyAika)/(60**2) ))

    
#%% Kagglen test
    
root_test = './input/vehicle/test/testset/'
data_test = []
id_line = []

iddi = 0
for file in sorted(os.listdir(os.path.join(root_test))):
    data_test.append((os.path.join(root_test, file)))
    id_line.append(iddi)
    iddi += 1

df_test = pd.DataFrame(data_test, columns=['file_path'])

test_datagen = ImageDataGenerator(rescale = 1./255)

#%% InceptionV3 TEST

testi_askel = 2

test_generator = test_datagen.flow_from_dataframe(
        dataframe = df_test,
        x_col = 'file_path',
        y_col = None,
        batch_size = testi_askel,
        shuffle = False,
        class_mode = None,
        target_size = (299,299)
        )

askelia = len(df_test) // testi_askel

nimi = "05_IncV3_acc_foldi_4.hdf5"
print("Ladataan verkko {}".format(nimi))
verkko = load_model(nimi, compile = False)
verkko.summary()
 
testia = np.array(verkko.predict_generator(test_generator,
                                steps = askelia,
                                verbose = 1,
                                workers = 7,
                                max_queue_size = 64))

K.clear_session()

for i in range(5,6):
        
    nimi = "05_IncV3_acc_foldi_{}.hdf5".format(i)
    print("Ladataan verkko: {}".format(nimi))
    verkko = load_model(nimi, compile = False)
    verkko.summary()
    
    test_generator.reset()
    
    pred = np.array(verkko.predict_generator(test_generator,
                                    steps = askelia,
                                    verbose = 1,
                                    workers = 7,
                                    max_queue_size = 64))
    testia = np.hstack((testia, pred))
    
    K.clear_session()    


#%% EffiNetB2 TEST
test_generator = test_datagen.flow_from_dataframe(
        dataframe = df_test,
        x_col = 'file_path',
        y_col = None,
        batch_size = testi_askel,
        shuffle = False,
        class_mode = None,
        target_size = (260,260)
        )
    
for i in range(2,4):
    
    nimi = "01_EffiNetB2_acc_foldi_{}.hdf5".format(i)
    print("Ladataan verkko {}".format(nimi))
    verkko = load_model(nimi, compile = False)
    verkko.summary()
    

    
    test_generator.reset()
     
    pred = np.array(verkko.predict_generator(test_generator,
                                    steps = askelia,
                                    verbose = 1,
                                    workers = 7,
                                    max_queue_size = 64))
    
    
    testia = np.hstack((testia, pred))
    
    K.clear_session()
        
#%% EffiNetB3 TEST

test_generator = test_datagen.flow_from_dataframe(
        dataframe = df_test,
        x_col = 'file_path',
        y_col = None,
        batch_size = testi_askel,
        shuffle = False,
        class_mode = None,
        target_size = (300,300)
        )

for i in range(0,2):

    nimi = "01_EffiNetB3_acc_foldi_{}.hdf5".format(i)
    print("Ladataan verkko {}".format(nimi))
    verkko = load_model(nimi, compile = False)
    verkko.summary()
  
    test_generator.reset()
    
    pred = np.array(verkko.predict_generator(test_generator,
                                    steps = askelia,
                                    verbose = 1,
                                    workers = 7,
                                    max_queue_size = 64))
    
    
    testia = np.hstack((testia, pred))
    
    K.clear_session()


#%% Kaggleen
     
pred_y = yhdistelija.predict(testia)
pred_y = le.inverse_transform(pred_y)
        
ulos = pd.DataFrame({'Id' : id_line, 'Category' : pred_y})
ulos.to_csv("yhdistelija_montaEri_SVC.csv", index = False)

print("Aikaa meni yhteensä {:5.3f} h".format( (time.time() - kokoaika)/(60**2) ))

    

