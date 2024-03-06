##### import glob
import shutil
import os
from multiprocessing import Pool
from contextlib import closing
import subprocess
import pandas as pd
import time
import logging
import logging.handlers
import os
import sys
import numpy as np
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
#import rawpy
from PIL import Image
import imageio
import cv2
import json 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
import time
from joblib import dump, load
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import tensorflow_probability as tfp
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"]="0"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# print(device_lib.list_local_devices())
def download_profiles(data):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger_file_handler = logging.handlers.RotatingFileHandler(
        "status_log_for_random_training.log",
        maxBytes=1024 * 1024,
        backupCount=1,
        encoding="utf8",
    )
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger_file_handler.setFormatter(formatter)
    logger.addHandler(logger_file_handler)

    for i in data.keys():
        #try:
        os.makedirs(data[i][0])
        subprocess.call(['chmod', '-R', '777', data[i][0]])
        '''except:
            print('skipping ', data[i][0])
            return''' 
        
        os.makedirs(data[i][0] + '/TIFFs', exist_ok=True)
        subprocess.call(['chmod', '-R', '777', data[i][0] + '/TIFFs'])

        cmd = '../google-cloud-sdk/bin/gsutil -m cp -r gs://editing_userdata/' + i + '/' + data[i][0] + '/training_data/Dataset/*/TIFFs/*.tif' + ' ' + data[i][0] + '/TIFFs'
        subprocess.call(cmd.split(' '))   
        print('Tiffs downloaded!')

        cmd = '../google-cloud-sdk/bin/gsutil -m cp -r gs://editing_userdata/' + i + '/' + data[i][0] + '/' + data[i][1] + '/trained_models' + ' ' + data[i][0]
        subprocess.call(cmd.split(' '))
        print('Training data downloaded!')

        latest_folder = sorted(os.listdir(data[i][0] + '/trained_models'))[-1]
        cmd = ['unzip', 
               '-o',
               data[i][0] + '/trained_models/' + latest_folder + '/debug.zip' , 
               '-d',  
               data[i][0] + '/trained_models/' + latest_folder]
        subprocess.call(cmd)
        print('Got slider exif')

def get_split_track(path):
    with open(path) as json_file:
        split_track = json.load(json_file)
    return split_track

def remove_out(data):
    Q1 = np.percentile(data['Temperature'], 5, interpolation = 'midpoint')
    Q3 = np.percentile(data['Temperature'], 95, interpolation = 'midpoint')

    IQR = Q3 - Q1
    upper = (Q3+1.5*IQR)
    lower = (Q1-1.5*IQR)
    print("Upper bound:", upper)
    print("Lower bound:", lower)
    print('OG Rows', data.shape[0])

    data = data[(lower <= data['Temperature']) & (data['Temperature'] <= upper)]
    print('Rem Rows:', data.shape[0])

    Q1 = np.percentile(data['Tint'], 5, interpolation = 'midpoint')
    Q3 = np.percentile(data['Tint'], 95, interpolation = 'midpoint')

    IQR = Q3 - Q1
    upper = (Q3+1.5*IQR)
    lower = (Q1-1.5*IQR)
    print("Upper bound:",upper)
    print("Lower bound:", lower)
    print('OG Rows', data.shape[0])

    data = data[(lower <= data['Tint']) & (data['Tint'] <= upper)]
    print('Rem Rows:', data.shape[0])

    return data

def get_splits(data, split_track, data_dir,x_factors,y_props):
    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []
    c, c1 = 0, 0
    for i in list(split_track.keys()):
        try:
            name = data_dir + i.split('/')[-1]
            
            if split_track[i]['veryclose'] == 'x_train':
                x_train.append(data.loc[data['img_path'] == name][['img_path']+x_factors].values[0])  
                y_train.append(data.loc[data['img_path'] == name][y_props].values[0])

            elif split_track[i]['veryclose'] == 'x_test':
                x_test.append(data.loc[data['img_path'] == name][['img_path']+x_factors].values[0])
                y_test.append(data.loc[data['img_path'] == name][y_props].values[0])

            elif split_track[i]['veryclose'] == 'x_val':
                x_val.append(data.loc[data['img_path'] == name][['img_path']+x_factors].values[0])
                y_val.append(data.loc[data['img_path'] == name][y_props].values[0])
            c+=1
        except:
            c1+=1
        
    print(c, c1)

    x_train, x_val, x_test = np.array(x_train), np.array(x_val),np.array(x_test)
    y_train, y_val, y_test = np.array(y_train), np.array(y_val),np.array(y_test)

    return x_train, x_val, x_test, y_train, y_val, y_test

def get_RGB(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_array = np.array(img)
    
    R = np.mean(img_array[:,:,0]/255)
    G = np.mean(img_array[:,:,1]/255)
    B = np.mean(img_array[:,:,2]/255)
    
    return R, G, B

def get_H(image_path):
    img = cv2.imread(image_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    H, S, V = cv2.split(img_hsv)
    H = H/255
    S = S/255
    V = V/255

    mean_H = np.mean(H)
    mean_S = np.mean(S)

    return mean_H

def get_S(image_path):
    img = cv2.imread(image_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    H, S, V = cv2.split(img_hsv)
    H = H/255
    S = S/255
    V = V/255

    mean_H = np.mean(H)
    mean_S = np.mean(S)

    return mean_S

def get_luminance(x):
    vR, vG, vB = get_RGB(x)
    Y = (0.2126 * sRGBtoLin(vR) + 0.7152 * sRGBtoLin(vG) + 0.0722 * sRGBtoLin(vB))
    return Y

def get_image_intensity(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    avg_intensity = np.mean(np.ravel(image))
    return avg_intensity
    
def sRGBtoLin(colorChannel):
    if colorChannel <= 0.04045:
        return colorChannel / 12.92
    else:
        return ((colorChannel + 0.055) / 1.055) ** 2.4

def YtoLstar(Y):
    if Y <= (216/24389):
        return Y * (24389/27)
    else:
        return (Y ** (1/3)) * 116 - 16


def train_Histogram_ExternalFeatures():
    for i in os.listdir('../tatumreid'):
        if os.path.isdir(i) and i[0] != '.':
            # print(i)
            res_dict = {'profile_id': [],'Temp MAE new':[], 'Tint MAE new':[], 'Temp R2 new':[], 'Tint R2 new':[]}
            
            data_dir = i + '/TIFFs/'
            latest_folder = sorted(os.listdir(i + '/trained_models'))[-1]
            
            sliders = pd.read_csv(i + '/trained_models/' + latest_folder + '/sliders_final.csv')
            print('length of data : ', len(sliders))
            
            split_track = get_split_track(i + '/trained_models/' + latest_folder + '/split_tracker.json')

            sliders['img_path'] = sliders['img_path'].apply(lambda x: data_dir + x.split('/')[-1])
            sliders['Hue'] = sliders['img_path'].apply(lambda x : get_H(x))
            sliders['Saturation'] = sliders['img_path'].apply(lambda x : get_S(x))
            sliders["Luminance"] = sliders["img_path"].apply(lambda x: get_luminance(x))

            props = []
            y_props = []
            x_factors = ["Hue", "Saturation", "Luminance", "isoSpeedRating", "shutterSpeed", "aperture", "flashFired", "focalLength"]
                         #"Blacks", "Contrast", "Temperature", "Exposure", "Highlights", "Shadows", "Whites"]
            # og_props = ['HueAdjustmentAqua', 'HueAdjustmentBlue', 'HueAdjustmentGreen', 'HueAdjustmentMagenta', 'HueAdjustmentOrange', 'HueAdjustmentPurple', 'HueAdjustmentRed', 'HueAdjustmentYellow']
            
            og_props = []
            for i in sliders.columns:
                if 'hueadjustment' in i.lower():
                    og_props.append(i)

            print("Initial Filters: {}".format(og_props))       
            for prop in og_props:
                db_name = f'{prop}'
                freq = sliders[db_name].value_counts().values[0]
                total = sliders.shape[0]

                if freq/total >= 0.95:
                    # print(sliders[db_name].value_counts().values)
                    # print("{} feature is removed. ".format(prop))
                    # print(db_name, freq, np.unique(sliders[db_name].to_numpy(), return_counts = True))
                    continue
                else:
                    props.append(prop)
                    y_props.append(db_name)

            if len(y_props) == 0:
                print('Nothing left to train! Exiting...')
                exit(0)

            print(f'Model will only be trained for: {props}')
            print(f'Final props: {props}')
            print(f'Y props: {y_props}')
            print(f'X factors: {x_factors}')

            X = sliders[['img_path'] + x_factors].values
            Y = sliders[y_props].values

            x_train, x_val, x_test, y_train, y_val, y_test = get_splits(sliders, split_track, data_dir,x_factors,y_props)

            num_train = len(x_train)
            num_val = len(x_val)
            num_test = len(x_test)

            print('Number of train images:', num_train)
            print('Number of val images:', num_val)
            print('Number of test images:', num_test)

            train_batch_size = 32
            val_batch_size = 32
            test_batch_size = 32

            if num_train % train_batch_size==0:
                train_steps = num_train//train_batch_size
            else:
                train_steps = num_train//train_batch_size + 1

            if num_val % val_batch_size==0:
                val_steps = num_val//val_batch_size
            else:
                val_steps = num_val//val_batch_size + 1

            print('Number of train steps:', train_steps)
            print('Number of val steps:', val_steps)

            def return_hist_rgb(img_numpy):
                r_hist = cv2.calcHist([img_numpy[:, :, 0]], [0], None, [256], [0, 256])
                g_hist = cv2.calcHist([img_numpy[:, :, 1]], [0], None, [256], [0, 256])
                b_hist = cv2.calcHist([img_numpy[:, :, 2]], [0], None, [256], [0, 256])
                merged_histogram = np.concatenate([r_hist, g_hist, b_hist], axis = -1)
                return r_hist, g_hist, b_hist, merged_histogram

            def parse_image_with_histogram(img_path, extradata, labels):
                img_path = img_path.numpy().decode('utf-8')
                image = np.array(Image.open(img_path))
                r_hist, g_hist, b_hist, merged_hist = return_hist_rgb(image)
                
                image = tf.cast(image, tf.float32)
                labels = tf.cast(labels, tf.float32)
                image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
                merged_hist = tf.cast(merged_hist, tf.float32)
                
                return image, extradata, labels, merged_hist

            def restore_inputs_with_histogram(image, extradata, labels, hist):
                inputs = {}
                inputs['image'] = image
                inputs['extradata'] = extradata
                inputs["histogram"] = hist
                return inputs, labels

            def set_shapes_with_histogram(image, data, labels, hist):
                image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                data.set_shape([num_factors, ])
                labels.set_shape([num_labels, ])
                hist.set_shape([256, 3])
                return image, data, labels, hist
                
            data_augmentation = tf.keras.Sequential([
              RandomFlip("horizontal"),
              # RandomRotation(0.2),
            ])


            AUTO = tf.data.AUTOTUNE
            IMAGE_SIZE = 256
            num_factors = len(x_factors)
            num_labels = len(y_props)


            train_ds = tf.data.Dataset.from_tensor_slices((x_train[:, 0], x_train[:, 1:].astype(np.float32), y_train))
            train_ds = train_ds.map(lambda x,y,z: tf.py_function(func=parse_image_with_histogram, inp=[x, y, z], Tout=[tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=AUTO, deterministic=False)
            train_ds = train_ds.map(lambda x,y,z,a: set_shapes_with_histogram(x, y, z, a)).cache("train").shuffle(500).repeat().batch(train_batch_size)
            train_ds = train_ds.map(lambda x,y,z,a: (data_augmentation(x, training=True), y, z, a), num_parallel_calls=AUTO)
            train_ds = train_ds.map(lambda x,y,z,a: restore_inputs_with_histogram(x, y, z, a)).prefetch(AUTO)

            val_ds = tf.data.Dataset.from_tensor_slices((x_val[:, 0], x_val[:, 1:].astype(np.float32), y_val))
            val_ds = val_ds.map(lambda x,y,z: tf.py_function(func=parse_image_with_histogram, inp=[x, y, z], Tout=[tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=AUTO)
            val_ds = val_ds.map(lambda x,y,z,a: restore_inputs_with_histogram(x, y, z, a))
            val_ds = val_ds.cache().batch(val_batch_size).prefetch(AUTO)

            test_ds = tf.data.Dataset.from_tensor_slices((x_test[:, 0], x_test[:, 1:].astype(np.float32), y_test))
            test_ds = test_ds.map(lambda x,y,z: tf.py_function(func=parse_image_with_histogram, inp=[x, y, z], Tout=[tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=AUTO)
            test_ds = test_ds.map(lambda x,y,z,a: restore_inputs_with_histogram(x, y, z, a))
            test_ds = test_ds.cache().batch(test_batch_size).prefetch(AUTO)
            
            image_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image')
            extra_input = tf.keras.layers.Input(shape=(num_factors,), name='extradata')
            hist_input = tf.keras.layers.Input(shape = (IMAGE_SIZE, 3), name = "histogram")

            base_model = tf.keras.applications.MobileNetV3Large(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                           include_top=False,
                                                           weights='imagenet')

            base_model.trainable = True
            base_model_output = base_model(image_input)
            image_embedding = tf.keras.layers.GlobalAveragePooling2D()(base_model_output)

            hist_layer_1 = tf.keras.layers.Conv1D(filters = 9, kernel_size = 16, strides = 8, padding = "valid", activation = "relu")(hist_input)
            hist_layer_2 = tf.keras.layers.Conv1D(filters = 1 ,kernel_size = 8, strides = 4, padding = "valid", activation = "relu")(hist_layer_1)
            hist_flatten = tf.keras.layers.Flatten()(hist_layer_1)
            
            all_features = tf.keras.layers.concatenate([image_embedding, extra_input, hist_flatten])
            model_output = tf.keras.layers.Dense(num_labels, dtype=tf.float32)(all_features)

            model = tf.keras.Model(inputs=[image_input, extra_input, hist_input], outputs=model_output)

            model.compile(
                    loss = tf.keras.losses.MeanAbsoluteError(),
                    optimizer = tf.keras.optimizers.Adam(),
                    metrics = [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
                    # steps_per_execution=64
                )

            model.summary()


            filepath = "hue_hist_ext_ft.h5"
            if not(os.path.exists(filepath)):

                checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, 
                                             save_best_only=True, mode='min')

                reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5, patience=2, 
                                                   verbose=1, mode='min', min_lr=0.0000000001)

                early = EarlyStopping(monitor='val_mean_absolute_error', verbose=1, mode='min', patience=8)

                callbacks_list = [checkpoint, reduce_lr, early]

                history = model.fit(train_ds, steps_per_epoch=train_steps, 
                                    validation_data=val_ds,
                                    validation_steps=val_steps,
                                    epochs=70, verbose=0,
                                    callbacks=callbacks_list)

            model = tf.keras.models.load_model(filepath)
            model.summary()
            
            model = tf.keras.models.load_model('hue_hist_ext_ft.h5')

            model.evaluate(val_ds)
            model.evaluate(test_ds)

            st = time.time()
            preds = model.predict(test_ds, verbose=1)
            et = time.time()
            print('Time took:', et-st)

            metrics = {'MAE': {}, 'R2': {}}


            for i, prop in enumerate(props):
                error = mae(y_test[:, i], preds[:, i])
                score = r2_score(y_test[:, i], preds[:, i])

                print(f'MAE for {prop}: {error}')
                print(f'R2 Score for {prop}: {score}')

                plt.figure(figsize=(20, 9))
                plt.plot(y_test[:, i])
                plt.plot(preds[:,i])
                plt.savefig(prop+'_new.png')    
                plt.clf()
                metrics['MAE'][prop] = error
                metrics['R2'][prop] = score
            print('NEW Metrics')
            print(metrics)


def train_Histogram_ProminentColor_Histogram_ExternalFeatures():
    for i in os.listdir('../tatumreid'):
        if os.path.isdir(i) and i[0] != '.':
            # print(i)
            res_dict = {'profile_id': [],'Temp MAE new':[], 'Tint MAE new':[], 'Temp R2 new':[], 'Tint R2 new':[]}
            
            data_dir = i + '/TIFFs/'
            latest_folder = sorted(os.listdir(i + '/trained_models'))[-1]
            
            sliders = pd.read_csv(i + '/trained_models/' + latest_folder + '/sliders_final.csv')
            print('length of data : ', len(sliders))
            
            split_track = get_split_track(i + '/trained_models/' + latest_folder + '/split_tracker.json')

            N_CLUSTERS= 16
            sliders['img_path'] = sliders['img_path'].apply(lambda x: data_dir + x.split('/')[-1])
            sliders['Hue'] = sliders['img_path'].apply(lambda x : get_H(x))
            sliders['Saturation'] = sliders['img_path'].apply(lambda x : get_S(x))
            sliders["Luminance"] = sliders["img_path"].apply(lambda x: get_luminance(x))
            # sliders["ev"] = get_ev(sliders["shutterSpeed"], sliders["isoSpeedRating"], sliders["aperture"])
            
            props = []
            y_props = []
            x_factors = ["Hue", "Saturation", "Luminance", "isoSpeedRating", "shutterSpeed", "aperture", "flashFired", "focalLength"]
            x_factors =[]
            
            
            og_props = []
            for i in sliders.columns:
                if 'hueadjustment' in i.lower():
                    og_props.append(i)

            print("Initial Filters: {}".format(og_props))       
            for prop in og_props:
                db_name = f'{prop}'
                freq = sliders[db_name].value_counts().values[0]
                total = sliders.shape[0]

                if freq/total >= 0.95:
                    # print(sliders[db_name].value_counts().values)
                    # print("{} feature is removed. ".format(prop))
                    # print(db_name, freq, np.unique(sliders[db_name].to_numpy(), return_counts = True))
                    continue
                else:
                    props.append(prop)
                    y_props.append(db_name)

            if len(y_props) == 0:
                print('Nothing left to train! Exiting...')
                exit(0)

            print(f'Model will only be trained for: {props}')
            print(f'Final props: {props}')
            print(f'Y props: {y_props}')
            print(f'X factors: {x_factors}')

            X = sliders[['img_path'] + x_factors].values
            Y = sliders[y_props].values

            x_train, x_val, x_test, y_train, y_val, y_test = get_splits(sliders, split_track, data_dir,x_factors,y_props)

            num_train = len(x_train)
            num_val = len(x_val)
            num_test = len(x_test)

            print('Number of train images:', num_train)
            print('Number of val images:', num_val)
            print('Number of test images:', num_test)

            train_batch_size = 32
            val_batch_size = 32
            test_batch_size = 32

            if num_train % train_batch_size==0:
                train_steps = num_train//train_batch_size
            else:
                train_steps = num_train//train_batch_size + 1

            if num_val % val_batch_size==0:
                val_steps = num_val//val_batch_size
            else:
                val_steps = num_val//val_batch_size + 1

            print('Number of train steps:', train_steps)
            print('Number of val steps:', val_steps)

            def return_hist_rgb(img_numpy):
                r_hist = cv2.calcHist([img_numpy[:, :, 0]], [0], None, [256], [0, 256])
                g_hist = cv2.calcHist([img_numpy[:, :, 1]], [0], None, [256], [0, 256])
                b_hist = cv2.calcHist([img_numpy[:, :, 2]], [0], None, [256], [0, 256])
                merged_histogram = np.concatenate([r_hist, g_hist, b_hist], axis = -1)
                return r_hist, g_hist, b_hist, merged_histogram

            def get_prominent_color(img):
                Z = img.reshape((-1,3))
                Z = np.float32(Z)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 70, 0.5)
                def colorQuant(Z, K, criteria):
                   ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
                   center = np.uint8(center)
                   res = center[label.flatten()]
                   res2 = res.reshape((img.shape))
                   unique_frequency = np.unique(label.flatten(), return_counts = True)
                   return res2, center, label.flatten(), unique_frequency
                res3, centers, labels , label_flatten= colorQuant(Z, 16, criteria)  
                
                total_sum  = np.sum(label_flatten[1]).item()
                prob = label_flatten[1]/total_sum

                
                center_prob = []
                for i in range(len(prob)):
                    center_prob.append(np.array([centers[i].tolist()+[prob[i]]], dtype = np.float32))
                center_prob = np.concatenate(center_prob, axis = 0)
                
                center_x = []
                for i in range(len(prob)):
                    center_x.append(np.array([centers[i].tolist()], dtype = np.float32))
                center_x = np.concatenate(center_x, axis = 0)
                return center_prob.reshape(-1), center_x.reshape(-1)


            def parse_image_with_histogram(img_path, extradata, labels):
                img_path = img_path.numpy().decode('utf-8')
                image = np.array(Image.open(img_path).convert("RGB"))

                prominent_color_with_prob, prominent_pixels = get_prominent_color(image)
                
                r_hist, g_hist, b_hist, merged_hist = return_hist_rgb(image)
                
                image = tf.cast(image, tf.float32)
                labels = tf.cast(labels, tf.float32)
                image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
                merged_hist = tf.cast(merged_hist, tf.float32)
                prominent_pixels = tf.cast(prominent_pixels, tf.float32)

                return image, extradata, labels, merged_hist, prominent_pixels

            def restore_inputs_with_histogram(image, extradata, labels, hist, prominent_pixels):
                inputs = {}
                inputs['image'] = image
                inputs['extradata'] = extradata
                inputs["histogram"] = hist
                inputs["prominent_pixels"] = prominent_pixels
                
                return inputs, labels

            def set_shapes_with_histogram(image, data, labels, hist, prominent_colors):
                image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                data.set_shape([num_factors, ])
                labels.set_shape([num_labels, ])
                hist.set_shape([256, 3])
                prominent_colors.set_shape([N_CLUSTERS*3, ])
                
                return image, data, labels, hist, prominent_colors
                
            data_augmentation = tf.keras.Sequential([
              RandomFlip("horizontal"),
              # RandomRotation(0.2),
            ])


            AUTO = tf.data.AUTOTUNE
            IMAGE_SIZE = 256
            num_factors = len(x_factors)
            num_labels = len(y_props)


            train_ds = tf.data.Dataset.from_tensor_slices((x_train[:, 0], x_train[:, 1:].astype(np.float32), y_train))
            train_ds = train_ds.map(lambda x,y,z: tf.py_function(func=parse_image_with_histogram, inp=[x, y, z], Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=AUTO, deterministic=False)
            train_ds = train_ds.map(lambda x,y,z,a,b: set_shapes_with_histogram(x, y, z, a, b)).cache("train").shuffle(500).repeat().batch(train_batch_size)
            train_ds = train_ds.map(lambda x,y,z,a,b: (data_augmentation(x, training=True), y, z, a, b), num_parallel_calls=AUTO)
            train_ds = train_ds.map(lambda x,y,z,a,b: restore_inputs_with_histogram(x, y, z, a, b)).prefetch(AUTO)

            val_ds = tf.data.Dataset.from_tensor_slices((x_val[:, 0], x_val[:, 1:].astype(np.float32), y_val))
            val_ds = val_ds.map(lambda x,y,z: tf.py_function(func=parse_image_with_histogram, inp=[x, y, z], Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=AUTO)
            val_ds = val_ds.map(lambda x,y,z,a,b: restore_inputs_with_histogram(x, y, z, a, b))
            val_ds = val_ds.cache().batch(val_batch_size).prefetch(AUTO)

            test_ds = tf.data.Dataset.from_tensor_slices((x_test[:, 0], x_test[:, 1:].astype(np.float32), y_test))
            test_ds = test_ds.map(lambda x,y,z: tf.py_function(func=parse_image_with_histogram, inp=[x, y, z], Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=AUTO)
            test_ds = test_ds.map(lambda x,y,z,a,b: restore_inputs_with_histogram(x, y, z, a, b))
            test_ds = test_ds.cache().batch(test_batch_size).prefetch(AUTO)
            # break
            image_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image')
            extra_input = tf.keras.layers.Input(shape=(num_factors,), name='extradata')
            hist_input = tf.keras.layers.Input(shape = (IMAGE_SIZE, 3), name = "histogram")
            prom_pixel_input = tf.keras.layers.Input(shape = (N_CLUSTERS*3, ), name = "prominent_pixels")

            base_model = tf.keras.applications.MobileNetV3Large(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                           include_top=False,
                                                           weights='imagenet')

            base_model.trainable = True
            base_model_output = base_model(image_input)
            image_embedding = tf.keras.layers.GlobalAveragePooling2D()(base_model_output)

            hist_layer_1 = tf.keras.layers.Conv1D(filters = 9, kernel_size = 16, strides = 8, padding = "valid", activation = "relu")(hist_input)
            hist_layer_2 = tf.keras.layers.Conv1D(filters = 1 ,kernel_size = 8, strides = 4, padding = "valid", activation = "relu")(hist_layer_1)
            hist_flatten = tf.keras.layers.Flatten()(hist_layer_1)        

            all_features = tf.keras.layers.concatenate([image_embedding, extra_input, hist_flatten, prom_pixel_input])
            model_output = tf.keras.layers.Dense(num_labels, dtype=tf.float32)(all_features)

            model = tf.keras.Model(inputs=[image_input, extra_input, hist_input, prom_pixel_input], outputs=model_output)

            model.compile(
                    loss = tf.keras.losses.MeanAbsoluteError(),
                    optimizer = tf.keras.optimizers.Adam(),
                    metrics = [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
                    # steps_per_execution=64
                )

            model.summary()


            filepath = "hue_hist_promcolor.h5"
            if not(os.path.exists(filepath)):

                checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, 
                                             save_best_only=True, mode='min')

                reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5, patience=2, 
                                                   verbose=1, mode='min', min_lr=0.0000000001)

                early = EarlyStopping(monitor='val_mean_absolute_error', verbose=1, mode='min', patience=8)

                callbacks_list = [checkpoint, reduce_lr, early]

                history = model.fit(train_ds, steps_per_epoch=train_steps, 
                                    validation_data=val_ds,
                                    validation_steps=val_steps,
                                    epochs=70, verbose=0,
                                    callbacks=callbacks_list)

            model = tf.keras.models.load_model(filepath)
            model.summary()
            
            model = tf.keras.models.load_model('hue_hist_promcolor.h5')

            model.evaluate(val_ds)
            model.evaluate(test_ds)

            st = time.time()
            preds = model.predict(test_ds, verbose=1)
            et = time.time()
            print('Time took:', et-st)

            metrics = {'MAE': {}, 'R2': {}}


            for i, prop in enumerate(props):
                error = mae(y_test[:, i], preds[:, i])
                score = r2_score(y_test[:, i], preds[:, i])

                print(f'MAE for {prop}: {error}')
                print(f'R2 Score for {prop}: {score}')

                plt.figure(figsize=(20, 9))
                plt.plot(y_test[:, i])
                plt.plot(preds[:,i])
                plt.savefig(prop+'_new.png')    
                plt.clf()
                metrics['MAE'][prop] = error
                metrics['R2'][prop] = score
            print('NEW Metrics')
            print(metrics)
            
def train_new():
    for i in os.listdir('../tatumreid'):
        if os.path.isdir(i) and i[0] != '.':
            print(i)
            res_dict = {'profile_id': [],'Temp MAE new':[], 'Tint MAE new':[], 'Temp R2 new':[], 'Tint R2 new':[]}
            
            data_dir = i + '/TIFFs/'
            latest_folder = sorted(os.listdir(i + '/trained_models'))[-1]
            
            sliders = pd.read_csv(i + '/trained_models/' + latest_folder + '/sliders_final.csv')
            print('length of data : ', len(sliders))
            
            split_track = get_split_track(i + '/trained_models/' + latest_folder + '/split_tracker.json')

            sliders['img_path'] = sliders['img_path'].apply(lambda x: data_dir + x.split('/')[-1])
            sliders['Hue'] = sliders['img_path'].apply(lambda x : get_H(x))
            sliders['Saturation'] = sliders['img_path'].apply(lambda x : get_S(x))
            sliders["Luminance"] = sliders["img_path"].apply(lambda x: get_luminance(x))
            # sliders["ev"] = get_ev(sliders["shutterSpeed"], sliders["isoSpeedRating"], sliders["aperture"])
            
            props = []
            y_props = []
            x_factors = ["Hue", "Saturation", "Luminance", "isoSpeedRating", "shutterSpeed", "aperture", "flashFired", "focalLength"]
            x_factors = []
            og_props = []
            for i in sliders.columns:
                if 'luminanceadjustment' in i.lower():
                    og_props.append(i)
                    
            for prop in og_props:
                db_name = f'{prop}'
                freq = sliders[db_name].value_counts().values[0]
                total = sliders.shape[0]

                if freq/total >= 0.95:
                    continue
                else:
                    props.append(prop)
                    y_props.append(db_name)

            if len(y_props) == 0:
                print('Nothing left to train! Exiting...')
                exit(0)

            print(f'Model will only be trained for: {props}')
            print(f'Final props: {props}')
            print(f'Y props: {y_props}')
            print(f'X factors: {x_factors}')

            X = sliders[['img_path'] + x_factors].values
            Y = sliders[y_props].values
            print(X)

            x_train, x_val, x_test, y_train, y_val, y_test = get_splits(sliders, split_track, data_dir,x_factors,y_props)

            num_train = len(x_train)
            num_val = len(x_val)
            num_test = len(x_test)

            print('Number of train images:', num_train)
            print('Number of val images:', num_val)
            print('Number of test images:', num_test)

            train_batch_size = 32
            val_batch_size = 32
            test_batch_size = 32

            if num_train % train_batch_size==0:
                train_steps = num_train//train_batch_size
            else:
                train_steps = num_train//train_batch_size + 1

            if num_val % val_batch_size==0:
                val_steps = num_val//val_batch_size
            else:
                val_steps = num_val//val_batch_size + 1

            print('Number of train steps:', train_steps)
            print('Number of val steps:', val_steps)

            def parse_image(img_path, extradata, labels):
                img_path = img_path.numpy().decode('utf-8')
                image = np.array(Image.open(img_path))

                image = tf.cast(image, tf.float32)
                labels = tf.cast(labels, tf.float32)
                image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
                return image, extradata, labels

            def restore_inputs(image, extradata, labels):
                inputs = {}
                inputs['image'] = image
                inputs['extradata'] = extradata
                return inputs, labels

            def set_shapes(image, data, labels):
                image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                data.set_shape([num_factors, ])
                labels.set_shape([num_labels, ])
                return image, data, labels

            data_augmentation = tf.keras.Sequential([
              RandomFlip("horizontal"),
              # RandomRotation(0.2),
            ])


            AUTO = tf.data.AUTOTUNE
            IMAGE_SIZE = 256
            num_factors = len(x_factors)
            num_labels = len(y_props)


            train_ds = tf.data.Dataset.from_tensor_slices((x_train[:, 0], x_train[:, 1:].astype(np.float32), y_train))
            train_ds = train_ds.map(lambda x,y,z: tf.py_function(func=parse_image, inp=[x, y, z], Tout=[tf.float32, tf.float32,             tf.float32]), num_parallel_calls=AUTO, deterministic=False)
            train_ds = train_ds.map(lambda x,y,z: set_shapes(x, y, z)).cache("train_ev").shuffle(500).repeat().batch(train_batch_size)
            train_ds = train_ds.map(lambda x,y,z: (data_augmentation(x, training=True), y, z), num_parallel_calls=AUTO)
            train_ds = train_ds.map(lambda x,y,z: restore_inputs(x, y, z)).prefetch(AUTO)

            val_ds = tf.data.Dataset.from_tensor_slices((x_val[:, 0], x_val[:, 1:].astype(np.float32), y_val))
            val_ds = val_ds.map(lambda x,y,z: tf.py_function(func=parse_image, inp=[x, y, z], Tout=[tf.float32, tf.float32,                 tf.float32]), num_parallel_calls=AUTO)
            val_ds = val_ds.map(lambda x,y,z: restore_inputs(x, y, z))
            val_ds = val_ds.cache().batch(val_batch_size).prefetch(AUTO)

            test_ds = tf.data.Dataset.from_tensor_slices((x_test[:, 0], x_test[:, 1:].astype(np.float32), y_test))
            test_ds = test_ds.map(lambda x,y,z: tf.py_function(func=parse_image, inp=[x, y, z], Tout=[tf.float32, tf.float32,               tf.float32]), num_parallel_calls=AUTO)
            test_ds = test_ds.map(lambda x,y,z: restore_inputs(x, y, z))
            test_ds = test_ds.cache().batch(test_batch_size).prefetch(AUTO)

            image_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image')
            extra_input = tf.keras.layers.Input(shape=(num_factors,), name='extradata')

            base_model = tf.keras.applications.MobileNetV3Large(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                           include_top=False,
                                                           weights='imagenet')

            base_model.trainable = True
            base_model_output = base_model(image_input)
            image_embedding = tf.keras.layers.GlobalAveragePooling2D()(base_model_output)
            all_features = tf.keras.layers.concatenate([image_embedding, extra_input])
            model_output = tf.keras.layers.Dense(num_labels, dtype=tf.float32)(all_features)

            model = tf.keras.Model(inputs=[image_input, extra_input], outputs=model_output)

            model.compile(
                    loss = tf.keras.losses.MeanAbsoluteError(),
                    optimizer = tf.keras.optimizers.Adam(),
                    metrics = [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
                    # steps_per_execution=64
                )

            model.summary()


            filepath = "luminance.h5"
            if not(os.path.exists(filepath)):

                checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, 
                                             save_best_only=True, mode='min')

                reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5, patience=2, 
                                                   verbose=1, mode='min', min_lr=0.0000000001)

                early = EarlyStopping(monitor='val_mean_absolute_error', verbose=1, mode='min', patience=8)

                callbacks_list = [checkpoint, reduce_lr, early]

                history = model.fit(train_ds, steps_per_epoch=train_steps, 
                                    validation_data=val_ds,
                                    validation_steps=val_steps,
                                    epochs=70, verbose=0,
                                    callbacks=callbacks_list)

            model = tf.keras.models.load_model(filepath)
            model.summary()
            
            model = tf.keras.models.load_model('luminance.h5')

            model.evaluate(val_ds)
            model.evaluate(test_ds)

            st = time.time()
            preds = model.predict(test_ds, verbose=1)
            et = time.time()
            print('Time took:', et-st)

            metrics = {'MAE': {}, 'R2': {}}


            for i, prop in enumerate(props):
                error = mae(y_test[:, i], preds[:, i])
                score = r2_score(y_test[:, i], preds[:, i])

                print(f'MAE for {prop}: {error}')
                print(f'R2 Score for {prop}: {score}')

                plt.figure(figsize=(20, 9))
                plt.plot(y_test[:, i])
                plt.plot(preds[:,i])
                plt.savefig(prop+'_new.png')    
                plt.clf()
                metrics['MAE'][prop] = error
                metrics['R2'][prop] = score
            print('NEW Metrics')
            print(metrics)




###################################################################################################

def train_Hist_HSLColor_Feature_Bank_archived():
    for i in os.listdir('../tatumreid'):
        if os.path.isdir(i) and i[0] != '.':

            res_dict = {'profile_id': [],'Temp MAE new':[], 'Tint MAE new':[], 'Temp R2 new':[], 'Tint R2 new':[]}
            
            data_dir = i + '/TIFFs/'
            latest_folder = sorted(os.listdir(i + '/trained_models'))[-1]
            
            sliders = pd.read_csv(i + '/trained_models/' + latest_folder + '/sliders_final.csv')
            print('length of data : ', len(sliders))
            
            split_track = get_split_track(i + '/trained_models/' + latest_folder + '/split_tracker.json')

            sliders['img_path'] = sliders['img_path'].apply(lambda x: data_dir + x.split('/')[-1])
            sliders['Hue'] = sliders['img_path'].apply(lambda x : get_H(x))
            sliders['Saturation'] = sliders['img_path'].apply(lambda x : get_S(x))
            sliders["Luminance"] = sliders["img_path"].apply(lambda x: get_luminance(x))

            props = []
            y_props = []
            x_factors = ["Hue", "Saturation", "Luminance", "isoSpeedRating", "shutterSpeed", "aperture", "flashFired", "focalLength"]
            x_factors = ["Hue", "Saturation", "Luminance", "isoSpeedRating", "shutterSpeed", "aperture"]
            
            og_props = []
            for i in sliders.columns:
                if 'hueadjustment' in i.lower():
                    og_props.append(i)

            print("Initial Filters: {}".format(og_props))       
            for prop in og_props:
                db_name = f'{prop}'
                freq = sliders[db_name].value_counts().values[0]
                total = sliders.shape[0]

                if freq/total >= 0.95:
                    # print(sliders[db_name].value_counts().values)
                    # print("{} feature is removed. ".format(prop))
                    # print(db_name, freq, np.unique(sliders[db_name].to_numpy(), return_counts = True))
                    continue
                else:
                    props.append(prop)
                    y_props.append(db_name)

            if len(y_props) == 0:
                print('Nothing left to train! Exiting...')
                exit(0)

            print(f'Model will only be trained for: {props}')
            print(f'Final props: {props}')
            print(f'Y props: {y_props}')
            print(f'X factors: {x_factors}')

            X = sliders[['img_path'] + x_factors].values
            Y = sliders[y_props].values

            x_train, x_val, x_test, y_train, y_val, y_test = get_splits(sliders, split_track, data_dir,x_factors,y_props)

            num_train = len(x_train)
            num_val = len(x_val)
            num_test = len(x_test)

            print('Number of train images:', num_train)
            print('Number of val images:', num_val)
            print('Number of test images:', num_test)

            train_batch_size = 32
            val_batch_size = 32
            test_batch_size = 32

            if num_train % train_batch_size==0:
                train_steps = num_train//train_batch_size
            else:
                train_steps = num_train//train_batch_size + 1

            if num_val % val_batch_size==0:
                val_steps = num_val//val_batch_size
            else:
                val_steps = num_val//val_batch_size + 1

            print('Number of train steps:', train_steps)
            print('Number of val steps:', val_steps)


            def get_lightroom_hsl_color_ranges():
                red_lower = np.array([0, 0, 0])
                red_upper = np.array([9, 255, 250])
                
                orange_lower = np.array([9, 0, 0])
                orange_upper = np.array([17, 255, 250])
                
                yellow_lower = np.array([13, 0, 0])
                yellow_upper = np.array([34, 255, 250])
                
                green_lower = np.array([34, 0, 0])
                green_upper = np.array([62, 255, 250])
                
                aqua_lower = np.array([62, 0, 0])
                aqua_upper = np.array([98, 255, 250])
                
                blue_lower = np.array([98, 0, 0])
                blue_upper = np.array([123, 255, 250])
                
                purple_lower = np.array([123, 0, 0])
                purple_upper = np.array([145, 255, 255])
                
                magenta_lower = np.array([145, 0, 0 ])
                magenta_upper = np.array([179, 255, 250])
            
                return red_lower, red_upper, orange_lower, orange_upper, yellow_lower, yellow_upper, green_lower, green_upper ,aqua_lower, aqua_upper, blue_lower, blue_upper, purple_lower, purple_upper, magenta_lower, magenta_upper

            r_l, r_u, o_l, o_u, y_l, y_u, g_l, g_u, a_l, a_u, b_l, b_u, p_l, p_u, m_l, m_u = get_lightroom_hsl_color_ranges()
            
            def generate_and_show_mask(hsv, rgb, color_lower, color_upper, plot_img = False):
                mask = cv2.inRange(hsv, color_lower, color_upper)
                result = cv2.bitwise_and(rgb, rgb, mask = mask)
                if plot_img == True:
                    plt.imshow(result)
                    plt.show()
                    plt.imshow(mask)
                    plt.show()
                return mask, result
                
            def return_hist_rgb(img_numpy):
                r_hist = cv2.calcHist([img_numpy[:, :, 0]], [0], None, [256], [0, 256])
                g_hist = cv2.calcHist([img_numpy[:, :, 1]], [0], None, [256], [0, 256])
                b_hist = cv2.calcHist([img_numpy[:, :, 2]], [0], None, [256], [0, 256])
                merged_histogram = np.concatenate([r_hist, g_hist, b_hist], axis = -1)
                return r_hist, g_hist, b_hist, merged_histogram

            def parse_image_with_histogram(img_path, extradata, labels):
                img_path = img_path.numpy().decode('utf-8')
                image = np.array(Image.open(img_path).convert("RGB"))
                hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

                red_mask, red_result = generate_and_show_mask(hsv_img, image, r_l, r_u)
                orange_mask, orange_result = generate_and_show_mask(hsv_img, image, o_l, o_u)
                yellow_mask, yellow_result = generate_and_show_mask(hsv_img, image, y_l, y_u)
                green_mask, green_result = generate_and_show_mask(hsv_img, image, g_l, g_u)
                aqua_mask, aqua_result = generate_and_show_mask(hsv_img, image, a_l, a_u)
                blue_mask, blue_result = generate_and_show_mask(hsv_img, image, b_l, b_u)
                purple_mask, purple_result = generate_and_show_mask(hsv_img, image, p_l, p_u)
                magenta_mask, magenta_result = generate_and_show_mask(hsv_img, image, m_l, m_u)
            
                r_hist, g_hist, b_hist, merged_hist = return_hist_rgb(image)
                
                image = tf.cast(image, tf.float32)
                labels = tf.cast(labels, tf.float32)
                image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
                merged_hist = tf.cast(merged_hist, tf.float32)
                
                red_result = tf.cast(red_result, tf.float32)
                orange_result = tf.cast(orange_result, tf.float32)
                yellow_result = tf.cast(yellow_result, tf.float32)
                green_result = tf.cast(green_result, tf.float32)
                aqua_result = tf.cast(aqua_result, tf.float32)
                blue_result = tf.cast(blue_result, tf.float32)
                purple_result = tf.cast(purple_result, tf.float32)
                magenta_result = tf.cast(magenta_result, tf.float32)
                
                return image, extradata, labels, merged_hist, red_result, orange_result,  yellow_result, green_result, aqua_result, blue_result, purple_result, magenta_result

            def restore_inputs_with_histogram(image, extradata, labels, hist, **kwargs):
                inputs = {}
                inputs['image'] = image
                inputs['extradata'] = extradata
                inputs["histogram"] = hist
                inputs['mask_red'] = kwargs["red"]
                inputs['mask_orange'] = kwargs["orange"]
                inputs['mask_yellow'] = kwargs["yellow"]
                inputs['mask_green'] = kwargs["green"]
                inputs['mask_aqua'] = kwargs["aqua"]
                inputs['mask_blue'] = kwargs["blue"]
                inputs['mask_purple'] = kwargs["purple"]
                inputs['mask_magenta'] = kwargs["magenta"]
                return inputs, labels

            def set_shapes_with_histogram(image, data, labels, hist, **kwargs):
                image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                data.set_shape([num_factors, ])
                labels.set_shape([num_labels, ])
                hist.set_shape([256, 3])

                kwargs["red"].set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                kwargs["orange"].set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                kwargs["yellow"].set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                kwargs["green"].set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                kwargs["aqua"].set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                kwargs["blue"].set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                kwargs["purple"].set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                kwargs["magenta"].set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                
                return image, data, labels, hist, kwargs
                
            data_augmentation = tf.keras.Sequential([
              RandomFlip("horizontal"),
              RandomRotation(0.2),
            ])


            AUTO = tf.data.AUTOTUNE
            IMAGE_SIZE = 256
            num_factors = len(x_factors)
            num_labels = len(y_props)


            train_ds = tf.data.Dataset.from_tensor_slices((x_train[:, 0], x_train[:, 1:].astype(np.float32), y_train))
            train_ds = train_ds.map(lambda x,y,z: tf.py_function(func=parse_image_with_histogram, inp=[x, y, z], Tout=[tf.float32, tf.float32, tf.float32, tf.float32,tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=AUTO, deterministic=False)
            train_ds = train_ds.map(lambda x,y,z,a,one, two, three, four, five, six, seven, eight: set_shapes_with_histogram(x, y, z, a, red = one, orange = two, yellow = three, green = four, aqua = five, blue = six, purple = seven, magenta = eight)).cache("train_hue").shuffle(500).repeat().batch(train_batch_size)
            train_ds = train_ds.map(lambda x,y,z,a, b: (data_augmentation(x, training=True), y, z, a, b), num_parallel_calls=AUTO)
            train_ds = train_ds.map(lambda x,y,z,a,b: restore_inputs_with_histogram(x, y, z, a, red = b["red"], orange = b["orange"], yellow = b["yellow"], green = b["green"], aqua = b["aqua"], blue= b["blue"], purple = b["purple"], magenta = b["magenta"])).prefetch(AUTO)

            
            val_ds = tf.data.Dataset.from_tensor_slices((x_val[:, 0], x_val[:, 1:].astype(np.float32), y_val))
            val_ds = val_ds.map(lambda x,y,z: tf.py_function(func=parse_image_with_histogram, inp=[x, y, z], Tout=[tf.float32, tf.float32, tf.float32, tf.float32 ,tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32 ]), num_parallel_calls=AUTO)
            val_ds = val_ds.map(lambda x,y,z,a,one, two, three, four, five, six, seven, eight: restore_inputs_with_histogram(x, y, z, a, red = one, orange = two, yellow = three, green = four, aqua = five, blue = six, purple = seven, magenta = eight))
            val_ds = val_ds.cache().batch(val_batch_size).prefetch(AUTO)

            test_ds = tf.data.Dataset.from_tensor_slices((x_test[:, 0], x_test[:, 1:].astype(np.float32), y_test))
            test_ds = test_ds.map(lambda x,y,z: tf.py_function(func=parse_image_with_histogram, inp=[x, y, z], Tout=[tf.float32, tf.float32, tf.float32, tf.float32 ,tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=AUTO)
            test_ds = test_ds.map(lambda x,y,z,a,one, two, three, four, five, six, seven, eight: restore_inputs_with_histogram(x, y, z, a, red = one, orange = two, yellow = three, green = four, aqua = five, blue = six, purple = seven, magenta = eight))
            test_ds = test_ds.cache().batch(test_batch_size).prefetch(AUTO)
            
            image_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image')

            red_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 3), name = "mask_red")
            orange_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 3), name = "mask_orange")
            yellow_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 3), name = 'mask_yellow')
            green_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 3), name = 'mask_green')
            aqua_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 3), name = 'mask_aqua')
            blue_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 3), name = 'mask_blue')
            purple_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 3), name = 'mask_purple')
            magenta_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 3), name = 'mask_magenta')
            
            extra_input = tf.keras.layers.Input(shape=(num_factors,), name='extradata')
            # hist_input = tf.keras.layers.Input(shape = (IMAGE_SIZE, 3), name = "histogram")

            base_model = tf.keras.applications.MobileNetV3Large(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                           include_top=False,
                                                           weights='imagenet')
            base_model.trainable = True
            # 
            mask_model = tf.keras.applications.MobileNetV3Small(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                                           include_top=False,
                                                           weights='imagenet')
            mask_model.trainable = False

            base_model_output = base_model(image_input)
            image_embedding = tf.keras.layers.GlobalAveragePooling2D()(base_model_output)

            # Mask Passing
            r_mask_embed = mask_model(red_mask_input)
            r_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(r_mask_embed)
            o_mask_embed = mask_model(orange_mask_input)
            o_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(o_mask_embed)
            y_mask_embed = mask_model(yellow_mask_input)
            y_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(y_mask_embed)
            g_mask_embed = mask_model(green_mask_input)
            g_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(g_mask_embed)
            a_mask_embed = mask_model(aqua_mask_input)
            a_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(a_mask_embed)
            b_mask_embed = mask_model(blue_mask_input)
            b_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(b_mask_embed)
            p_mask_embed = mask_model(purple_mask_input)
            p_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(p_mask_embed)
            m_mask_embed = mask_model(magenta_mask_input)
            m_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(m_mask_embed)

            avg_color_embed = tf.keras.layers.Average()([r_mask_embed,o_mask_embed, y_mask_embed, g_mask_embed, a_mask_embed, b_mask_embed, p_mask_embed, m_mask_embed])

            # hist_layer_1 = tf.keras.layers.Conv1D(filters = 9, kernel_size = 8, strides = 4, padding = "valid", activation = "relu")(hist_input)
            # hist_layer_2 = tf.keras.layers.Conv1D(filters = 1 ,kernel_size = 4, strides = 2, padding = "valid", activation = "relu")(hist_layer_1)
            # hist_flatten = tf.keras.layers.Flatten()(hist_layer_2)
            
            # all_features = tf.keras.layers.concatenate([image_embedding, extra_input, hist_flatten, avg_color_embed])
            all_features = tf.keras.layers.concatenate([image_embedding, extra_input, avg_color_embed])
            model_output = tf.keras.layers.Dense(num_labels, dtype=tf.float32)(all_features)

            model = tf.keras.Model(inputs=[image_input, extra_input, red_mask_input, orange_mask_input, yellow_mask_input, green_mask_input, aqua_mask_input, blue_mask_input, purple_mask_input, magenta_mask_input], outputs=model_output)
            # model = tf.keras.Model(inputs=[image_input, extra_input, hist_input, red_mask_input, orange_mask_input, yellow_mask_input, green_mask_input, aqua_mask_input, blue_mask_input, purple_mask_input, magenta_mask_input], outputs=model_output)

            model.compile(
                    loss = tf.keras.losses.MeanAbsoluteError(),
                    optimizer = tf.keras.optimizers.Adam(),
                    metrics = [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
                    # steps_per_execution=64
                )

            model.summary()


            filepath = "hue_ftbank_hist_colormask_small.h5"
            if not(os.path.exists(filepath)):

                checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, 
                                             save_best_only=True, mode='min')

                reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5, patience=2, 
                                                   verbose=1, mode='min', min_lr=0.0000000001)

                early = EarlyStopping(monitor='val_mean_absolute_error', verbose=1, mode='min', patience=8)

                callbacks_list = [checkpoint, reduce_lr, early]

                history = model.fit(train_ds, steps_per_epoch=train_steps, 
                                    validation_data=val_ds,
                                    validation_steps=val_steps,
                                    epochs=70, verbose=0,
                                    callbacks=callbacks_list)

            model = tf.keras.models.load_model(filepath)
            model.summary()
            
            model = tf.keras.models.load_model('hue_ftbank_hist_colormask_small.h5')

            model.evaluate(val_ds)
            model.evaluate(test_ds)

            st = time.time()
            preds = model.predict(test_ds, verbose=1)
            et = time.time()
            print('Time took:', et-st)

            metrics = {'MAE': {}, 'R2': {}}


            for i, prop in enumerate(props):
                error = mae(y_test[:, i], preds[:, i])
                score = r2_score(y_test[:, i], preds[:, i])

                print(f'MAE for {prop}: {error}')
                print(f'R2 Score for {prop}: {score}')

                plt.figure(figsize=(20, 9))
                plt.plot(y_test[:, i])
                plt.plot(preds[:,i])
                plt.savefig(prop+'_new.png')    
                plt.clf()
                metrics['MAE'][prop] = error
                metrics['R2'][prop] = score
            print('NEW Metrics')
            print(metrics)


def train_Hist_HSLColor_Feature_Bank():
    name_of_the_sliders = "hue"
    for i in os.listdir('/app/data'):
        if os.path.isdir(i) and i[0] != '.' and ".tf" not in i:
            # print(i)
            res_dict = {'profile_id': [],'Temp MAE new':[], 'Tint MAE new':[], 'Temp R2 new':[], 'Tint R2 new':[]}
            
            data_dir = i + '/TIFFs/'
            sliders = pd.read_csv(i + '/out' + '/sliders_final.csv')
            print('length of data : ', len(sliders))
            
            split_track = get_split_track(i + '/out' + '/split_tracker.json')

            def get_image_intensity(image):
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                avg_intensity = np.mean(np.ravel(image))
                return avg_intensity
            # sliders['img_path'] = sliders['img_path'].apply(lambda x: data_dir + x.split('/')[-1])
            sliders['Hue'] = sliders['img_path'].apply(lambda x : get_H(x))
            sliders['Saturation'] = sliders['img_path'].apply(lambda x : get_S(x))
            sliders["Luminance"] = sliders["img_path"].apply(lambda x: get_luminance(x))
            sliders["intensity"] = sliders["img_path"].apply(lambda x: get_image_intensity(x))

            props = []
            y_props = []
            x_factors = ["Hue", "Saturation", "Luminance", "isoSpeedRating", "shutterSpeed", "aperture", "flashFired", "focalLength", "intensity"]
            og_props = []
            for i in sliders.columns:
                if '{}adjustment'.format(name_of_the_sliders) in i.lower():
                    og_props.append(i)

            print("Initial Filters: {}".format(og_props))       
            for prop in og_props:
                db_name = f'{prop}'
                freq = sliders[db_name].value_counts().values[0]
                total = sliders.shape[0]
                if freq/total >= 0.95:
                    continue
                else:
                    props.append(prop)
                    y_props.append(db_name)

            if len(y_props) == 0:
                print('Nothing left to train! Exiting...')
                exit(0)

            print(f'Model will only be trained for: {props}')
            print(f'Final props: {props}')
            print(f'Y props: {y_props}')
            print(f'X factors: {x_factors}')

            X = sliders[['img_path'] + x_factors].values
            Y = sliders[y_props].values

            x_train, x_val, x_test, y_train, y_val, y_test = get_splits(sliders, split_track, data_dir,x_factors,y_props)

            num_train = len(x_train)
            num_val = len(x_val)
            num_test = len(x_test)

            print('Number of train images:', num_train)
            print('Number of val images:', num_val)
            print('Number of test images:', num_test)

            train_batch_size = 32
            val_batch_size = 32
            test_batch_size = 32

            if num_train % train_batch_size==0:
                train_steps = num_train//train_batch_size
            else:
                train_steps = num_train//train_batch_size + 1

            if num_val % val_batch_size==0:
                val_steps = num_val//val_batch_size
            else:
                val_steps = num_val//val_batch_size + 1

            print('Number of train steps:', train_steps)
            print('Number of val steps:', val_steps)


            def get_lightroom_hsl_color_ranges():
                red_lower = np.array([0, 0, 0])
                red_upper = np.array([9, 255, 250])
                
                orange_lower = np.array([9, 0, 0])
                orange_upper = np.array([17, 255, 250])
                
                yellow_lower = np.array([13, 0, 0])
                yellow_upper = np.array([34, 255, 250])
                
                green_lower = np.array([34, 0, 0])
                green_upper = np.array([62, 255, 250])
                
                aqua_lower = np.array([62, 0, 0])
                aqua_upper = np.array([98, 255, 250])
                
                blue_lower = np.array([98, 0, 0])
                blue_upper = np.array([123, 255, 250])
                
                purple_lower = np.array([123, 0, 0])
                purple_upper = np.array([145, 255, 255])
                
                magenta_lower = np.array([145, 0, 0 ])
                magenta_upper = np.array([179, 255, 250])
            
                return red_lower, red_upper, orange_lower, orange_upper, yellow_lower, yellow_upper, green_lower, green_upper ,aqua_lower, aqua_upper, blue_lower, blue_upper, purple_lower, purple_upper, magenta_lower, magenta_upper

            r_l, r_u, o_l, o_u, y_l, y_u, g_l, g_u, a_l, a_u, b_l, b_u, p_l, p_u, m_l, m_u = get_lightroom_hsl_color_ranges()
            
            def generate_and_show_mask(hsv, rgb, color_lower, color_upper, plot_img = False):
                mask = cv2.inRange(hsv, color_lower, color_upper)
                result = cv2.bitwise_and(rgb, rgb, mask = mask)
                result = cv2.resize(result, (224, 224))
                result = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
                if plot_img == True:
                    plt.imshow(result)
                    plt.show()
                    plt.imshow(mask)
                    plt.show()
                return mask, result
                
            def return_hist_rgb(img_numpy):
                r_hist = cv2.calcHist([img_numpy[:, :, 0]], [0], None, [256], [0, 256])
                g_hist = cv2.calcHist([img_numpy[:, :, 1]], [0], None, [256], [0, 256])
                b_hist = cv2.calcHist([img_numpy[:, :, 2]], [0], None, [256], [0, 256])
                merged_histogram = np.concatenate([r_hist, g_hist, b_hist], axis = -1)
                return r_hist, g_hist, b_hist, merged_histogram

            def parse_image_with_histogram(img_path, extradata, labels):
                img_path = img_path.numpy().decode('utf-8')
                image = np.array(Image.open(img_path).convert("RGB"))
                hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

                red_mask, red_result = generate_and_show_mask(hsv_img, image, r_l, r_u)
                orange_mask, orange_result = generate_and_show_mask(hsv_img, image, o_l, o_u)
                yellow_mask, yellow_result = generate_and_show_mask(hsv_img, image, y_l, y_u)
                green_mask, green_result = generate_and_show_mask(hsv_img, image, g_l, g_u)
                aqua_mask, aqua_result = generate_and_show_mask(hsv_img, image, a_l, a_u)
                blue_mask, blue_result = generate_and_show_mask(hsv_img, image, b_l, b_u)
                purple_mask, purple_result = generate_and_show_mask(hsv_img, image, p_l, p_u)
                magenta_mask, magenta_result = generate_and_show_mask(hsv_img, image, m_l, m_u)
            
                r_hist, g_hist, b_hist, merged_hist = return_hist_rgb(image)
                image = hsv_img
                image = tf.cast(image, tf.float32)
                labels = tf.cast(labels, tf.float32)
                image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
                merged_hist = tf.cast(merged_hist, tf.float32)
                
                red_result = tf.cast(red_result, tf.float32)
                orange_result = tf.cast(orange_result, tf.float32)
                yellow_result = tf.cast(yellow_result, tf.float32)
                green_result = tf.cast(green_result, tf.float32)
                aqua_result = tf.cast(aqua_result, tf.float32)
                blue_result = tf.cast(blue_result, tf.float32)
                purple_result = tf.cast(purple_result, tf.float32)
                magenta_result = tf.cast(magenta_result, tf.float32)
                return image, extradata, labels, merged_hist, red_result, orange_result,  yellow_result, green_result, aqua_result, blue_result, purple_result, magenta_result

            def restore_inputs_with_histogram(image, extradata, labels, hist, **kwargs):
                inputs = {}
                inputs['image'] = image
                inputs['extradata'] = extradata
                inputs["histogram"] = hist
                inputs['mask_red'] = kwargs["red"]
                inputs['mask_orange'] = kwargs["orange"]
                inputs['mask_yellow'] = kwargs["yellow"]
                inputs['mask_green'] = kwargs["green"]
                inputs['mask_aqua'] = kwargs["aqua"]
                inputs['mask_blue'] = kwargs["blue"]
                inputs['mask_purple'] = kwargs["purple"]
                inputs['mask_magenta'] = kwargs["magenta"]
                return inputs, labels

            def set_shapes_with_histogram(image, data, labels, hist, **kwargs):
                image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
                data.set_shape([num_factors, ])
                labels.set_shape([num_labels, ])
                hist.set_shape([256, 3])

                IMAGE_SIZE_1 = 224
                kwargs["red"].set_shape([IMAGE_SIZE_1, IMAGE_SIZE_1, 3])
                kwargs["orange"].set_shape([IMAGE_SIZE_1, IMAGE_SIZE_1, 3])
                kwargs["yellow"].set_shape([IMAGE_SIZE_1, IMAGE_SIZE_1, 3])
                kwargs["green"].set_shape([IMAGE_SIZE_1, IMAGE_SIZE_1, 3])
                kwargs["aqua"].set_shape([IMAGE_SIZE_1, IMAGE_SIZE_1, 3])
                kwargs["blue"].set_shape([IMAGE_SIZE_1, IMAGE_SIZE_1, 3])
                kwargs["purple"].set_shape([IMAGE_SIZE_1, IMAGE_SIZE_1, 3])
                kwargs["magenta"].set_shape([IMAGE_SIZE_1, IMAGE_SIZE_1, 3])
                
                return image, data, labels, hist, kwargs
                
            data_augmentation = tf.keras.Sequential([
              RandomFlip("horizontal"),
              RandomRotation(0.2),
                
            ])


            AUTO = tf.data.AUTOTUNE
            IMAGE_SIZE = 256
            num_factors = len(x_factors)
            num_labels = len(y_props)

            train_ds = tf.data.Dataset.from_tensor_slices((x_train[:, 0], x_train[:, 1:].astype(np.float32), y_train))
            train_ds = train_ds.map(lambda x,y,z: tf.py_function(func=parse_image_with_histogram, inp=[x, y, z], Tout=[tf.float32, tf.float32, tf.float32, tf.float32,tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=AUTO, deterministic=False)
            train_ds = train_ds.map(lambda x,y,z,a,one, two, three, four, five, six, seven, eight: set_shapes_with_histogram(x, y, z, a, red = one, orange = two, yellow = three, green = four, aqua = five, blue = six, purple = seven, magenta = eight)).cache("train_{}_loss".format(name_of_the_sliders)).shuffle(500).repeat().batch(train_batch_size)
            train_ds = train_ds.map(lambda x,y,z,a, b: (data_augmentation(x, training=True), y, z, a, b), num_parallel_calls=AUTO)
            train_ds = train_ds.map(lambda x,y,z,a,b: restore_inputs_with_histogram(x, y, z, a, red = b["red"], orange = b["orange"], yellow = b["yellow"], green = b["green"], aqua = b["aqua"], blue= b["blue"], purple = b["purple"], magenta = b["magenta"])).prefetch(AUTO)

            
            val_ds = tf.data.Dataset.from_tensor_slices((x_val[:, 0], x_val[:, 1:].astype(np.float32), y_val))
            val_ds = val_ds.map(lambda x,y,z: tf.py_function(func=parse_image_with_histogram, inp=[x, y, z], Tout=[tf.float32, tf.float32, tf.float32, tf.float32 ,tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32 ]), num_parallel_calls=AUTO)
            val_ds = val_ds.map(lambda x,y,z,a,one, two, three, four, five, six, seven, eight: restore_inputs_with_histogram(x, y, z, a, red = one, orange = two, yellow = three, green = four, aqua = five, blue = six, purple = seven, magenta = eight))
            val_ds = val_ds.cache().batch(val_batch_size).prefetch(AUTO)

            test_ds = tf.data.Dataset.from_tensor_slices((x_test[:, 0], x_test[:, 1:].astype(np.float32), y_test))
            test_ds = test_ds.map(lambda x,y,z: tf.py_function(func=parse_image_with_histogram, inp=[x, y, z], Tout=[tf.float32, tf.float32, tf.float32, tf.float32 ,tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]), num_parallel_calls=AUTO)
            test_ds = test_ds.map(lambda x,y,z,a,one, two, three, four, five, six, seven, eight: restore_inputs_with_histogram(x, y, z, a, red = one, orange = two, yellow = three, green = four, aqua = five, blue = six, purple = seven, magenta = eight))
            test_ds = test_ds.cache().batch(test_batch_size).prefetch(AUTO)

            
            ############################################
            image_input = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image')

            IMAGE_SIZE_EFFICIENT = 224
            red_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE_EFFICIENT, IMAGE_SIZE_EFFICIENT, 3), name = "mask_red")
            orange_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE_EFFICIENT, IMAGE_SIZE_EFFICIENT, 3), name = "mask_orange")
            yellow_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE_EFFICIENT, IMAGE_SIZE_EFFICIENT, 3), name = 'mask_yellow')
            green_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE_EFFICIENT, IMAGE_SIZE_EFFICIENT, 3), name = 'mask_green')
            aqua_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE_EFFICIENT, IMAGE_SIZE_EFFICIENT, 3), name = 'mask_aqua')
            blue_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE_EFFICIENT, IMAGE_SIZE_EFFICIENT, 3), name = 'mask_blue')
            purple_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE_EFFICIENT, IMAGE_SIZE_EFFICIENT, 3), name = 'mask_purple')
            magenta_mask_input = tf.keras.layers.Input(shape = (IMAGE_SIZE_EFFICIENT, IMAGE_SIZE_EFFICIENT, 3), name = 'mask_magenta')
            
            extra_input = tf.keras.layers.Input(shape=(num_factors,), name='extradata')
            # base_model = tf.keras.applications.MobileNetV3Large(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
            #                                                include_top=False,
            #                                                weights='imagenet')
            base_model = tf.keras.applications.regnet.RegNetY004(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),model_name='regnety004', include_top=False, weights='imagenet')
            # base_model.load_weights("mobilenetv3_base_weights.h5")
            base_model.trainable = True
            # 
            mask_model = tf.keras.applications.regnet.RegNetX002(input_shape=(IMAGE_SIZE_EFFICIENT, IMAGE_SIZE_EFFICIENT, 3), include_top=False, model_name='regnetx002', weights='imagenet')
            mask_model.trainable = True

            base_model_output = base_model(image_input)
            image_embedding = tf.keras.layers.GlobalAveragePooling2D()(base_model_output)

            r_mask_embed = mask_model(red_mask_input)
            r_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(r_mask_embed)
            r_mask_embed = tf.keras.layers.Dense(64, activation = "relu" , dtype=tf.float32)(r_mask_embed)
            o_mask_embed = mask_model(orange_mask_input)
            o_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(o_mask_embed)
            o_mask_embed = tf.keras.layers.Dense(64, activation = "relu" ,dtype=tf.float32)(o_mask_embed)
            y_mask_embed = mask_model(yellow_mask_input)
            y_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(y_mask_embed)
            y_mask_embed = tf.keras.layers.Dense(64, activation = "relu" ,dtype=tf.float32)(y_mask_embed)
            g_mask_embed = mask_model(green_mask_input)
            g_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(g_mask_embed)
            g_mask_embed = tf.keras.layers.Dense(64, activation = "relu" , dtype=tf.float32)(g_mask_embed)
            a_mask_embed = mask_model(aqua_mask_input)
            a_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(a_mask_embed)
            a_mask_embed = tf.keras.layers.Dense(64, activation = "relu" , dtype=tf.float32)(a_mask_embed)
            b_mask_embed = mask_model(blue_mask_input)
            b_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(b_mask_embed)
            b_mask_embed = tf.keras.layers.Dense(64,activation = "relu" , dtype=tf.float32)(b_mask_embed)
            p_mask_embed = mask_model(purple_mask_input)
            p_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(p_mask_embed)
            p_mask_embed = tf.keras.layers.Dense(64, activation = "relu" ,dtype=tf.float32)(p_mask_embed)
            m_mask_embed = mask_model(magenta_mask_input)
            m_mask_embed = tf.keras.layers.GlobalAveragePooling2D()(m_mask_embed)
            m_mask_embed = tf.keras.layers.Dense(64, activation = "relu" ,dtype=tf.float32)(m_mask_embed)

            img_extra_input = tf.keras.layers.concatenate([image_embedding, extra_input])
            img_dense_1 = tf.keras.layers.Dense(256, activation = "relu" , dtype = tf.float32)(img_extra_input)
            mask_concat = tf.keras.layers.concatenate([r_mask_embed,o_mask_embed, y_mask_embed, g_mask_embed, a_mask_embed, b_mask_embed, p_mask_embed, m_mask_embed])
            # mask_dense_1 = tf.keras.layers.Dense(256, dtype = tf.float32)(mask_concat)

            concat_1 = tf.keras.layers.concatenate([img_dense_1, mask_concat])
            
            model_output = tf.keras.layers.Dense(num_labels, dtype=tf.float32)(concat_1)

            model = tf.keras.Model(inputs=[image_input, extra_input, red_mask_input, orange_mask_input, yellow_mask_input, green_mask_input, aqua_mask_input, blue_mask_input, purple_mask_input, magenta_mask_input], outputs=model_output)
            
            ############################################
            

            def loss_fn_mae_rmse(y_true, y_pred, alpha=0.6):
                mae = tf.keras.losses.MeanAbsoluteError()
                mse = BMCLossMD_keras()
                # mse = BMCLossMD(init_noise_sigma = 1.0)
                return alpha * mae(y_true, y_pred) + (1 - alpha) * tf.sqrt(mse(y_true, y_pred))

            def bmc_loss_tf(y_pred, target, noise = 0.01):
                noise = noise ** 2
            
                I = tf.eye(8)
                # loc = np.expand_dims(y_pred, 1)
                loc = tf.expand_dims(y_pred, axis = 1)
                
                noise_var = noise
                var = noise_var*I
            
                mvn = tfp.distributions.MultivariateNormalFullCovariance(loc, var)
                target = tf.expand_dims(target, 0)
                # logits = mvn.log_prob(np.expand_dims(target, 0))
                logits = mvn.log_prob(target)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.range(0, train_batch_size, dtype=tf.int32))
                loss = tf.math.reduce_mean(loss, axis=0)
                
                loss  = loss * (2 * noise_var)
                return loss

            class BMCLossMD_keras(tf.keras.losses.Loss):
                def __init__(self, noise: float = 1.0):
                    super().__init__()
                    # self.noise = noise
                    self.noise = tf.Variable(initial_value=noise, dtype=tf.float32)
                    
                def call(self, y_true, y_pred):
                    noise = self.noise ** 2
                
                    I = tf.eye(y_pred.shape[-1])
                    loc = tf.expand_dims(y_pred, 1)
                    
                    noise_var = noise
                    var = noise_var*I
                
                    # mvn = tfp.distributions.MultivariateNormalFullCovariance(loc, var)
                    mvn = tfp.distributions.MultivariateNormalTriL(loc=loc, scale_tril=tf.linalg.cholesky(var))
                    logits = mvn.log_prob(tf.expand_dims(y_true, 0))

                    # logits.shape[0] and y_shape.shape[0] would be same (batch_size) 
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=tf.transpose(logits), 
                        # labels=tf.range(0, logits.shape[0], dtype=tf.int32)
                        labels=tf.range(0, tf.shape(logits)[0], dtype=tf.int32)
                    )
                    loss = tf.math.reduce_mean(loss, axis=0)
                    
                    loss  = loss * (2 * noise_var)
                    return loss
                
            model.compile(
                    loss = tf.keras.losses.MeanAbsoluteError(),
                    optimizer = tf.keras.optimizers.Adam(),
                    metrics = [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
                    # st
            model.summary()
                
            filepath = "{}_mae_both_regnet.tf".format(name_of_the_sliders)
            if not(os.path.exists(filepath)):

                checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, 
                                             save_best_only=True, mode='min')

                reduce_lr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5, patience=2, 
                                                   verbose=1, mode='min', min_lr=0.0000000001)

                early = EarlyStopping(monitor='val_mean_absolute_error', verbose=1, mode='min', patience=10)

                callbacks_list = [checkpoint, reduce_lr, early]

                history = model.fit(train_ds, steps_per_epoch=train_steps, 
                                    validation_data=val_ds,
                                    validation_steps=val_steps,
                                    epochs=70, verbose=1,
                                    callbacks=callbacks_list)

            # model = tf.keras.models.load_model(filepath, custom_objects= {"loss_fn_mae_rmse":loss_fn_mae_rmse })
            model = tf.keras.models.load_model(filepath)
            model.summary()
            
            model = tf.keras.models.load_model('{}_mae_both_regnet.tf'.format(name_of_the_sliders))

            ######################################################################

            def get_element_wise_distance(a, b):
                a = np.array(a)
                b = np.array(b)
                diff = a-b 
                diff = np.absolute(diff)
                diff = diff.sum()
                return diff

            def vector_distance_list_from_clusters(feat, clusters):
                list_of_pred =[]
                for k in clusters:
                    dist = get_element_wise_distance(feat, k)
                    list_of_pred.append([feat, k, dist])
                return list_of_pred
                
            class FeatureBank:
                def __init__(self, csv, y_feat):
                    self.csv = pd.read_csv(csv)
                    self.y_feat = y_feat
                    self.n_clusters = 32
                    self.sort_by_frequency = True
                    self.select_feat_numpy = self.csv[y_feat].values
                    self.select_feat = self.csv[y_feat].values.tolist()
                    self.list_of_arrays = [np.array(x, dtype = np.float32) for x in self.select_feat]
                    
                    self.clusters = self.get_non_zero_diff_clusters()
                    self.clusters_np = np.concatenate([np.array(x[0]).reshape(1, -1) for x in self.clusters], axis = 0)
                    self.k_means_clusters  =self.get_k_means_clusters()
                    self.k_means_clusters_np = np.concatenate([np.array(x).reshape(1, -1) for x in self.k_means_clusters], axis = 0)
            
                    if len(self.clusters)> len(self.k_means_clusters) and len(self.clusters)> self.n_clusters:
                        self.final_cluster_set = self.k_means_clusters
                    else:
                        self.final_cluster_set = self.clusters
            
                def get_k_means_clusters(self):
                    kmeans = KMeans(n_clusters = self.n_clusters)
                    kmeans.fit_predict(self.select_feat_numpy)
                    centers = kmeans.cluster_centers_
                    centers_int = centers.astype(np.int8).tolist()
                    return centers_int
                    
                def diff_hsl(self, ten1, ten2):
                    sub = (ten1==ten2).all()
                    if sub:
                        return True
                    else:
                        return False
            
                def get_non_zero_diff_clusters(self):
                    list_of_cluster_terms = []
                    running_point = None
                    for idx, x in enumerate(self.list_of_arrays):
                        if idx == 0:
                            running_point = x
                            list_of_cluster_terms.append([x, 1, [idx]])
                            continue
                        running_point = x
                        match_bool = False
                        for idx_int , unique_list in enumerate(list_of_cluster_terms):
                            unique_point, freq, indexes = unique_list
                            
                            if self.diff_hsl(running_point, unique_point) == True:
                                list_of_cluster_terms[idx_int][-2] +=1 
                                list_of_cluster_terms[idx_int][-1].append(idx)
                                match_bool = True
                                break
                            else:
                                continue
                                
                        if match_bool == False:
                            list_of_cluster_terms.append([running_point, 1, [idx]])
            
                    if self.sort_by_frequency:
                        list_of_cluster_terms = sorted(list_of_cluster_terms, key = lambda x: x[1], reverse = False)
                    
                    return list_of_cluster_terms    


            # feature_bank = FeatureBank(csv = i + '/trained_models/' + latest_folder + '/sliders_final.csv', y_feat = y_props)
            # np_cluster = feature_bank.clusters_np
            
            model.evaluate(val_ds)
            model.evaluate(test_ds)

            st = time.time()
            preds = model.predict(test_ds, verbose=1)
            et = time.time()
            print('Time took:', et-st)

            metrics = {'MAE': {}, 'R2': {}}

            def cal_cosine_sim(A, B):
                from numpy.linalg import norm
                return np.dot(A, B)/(norm(A)*norm(B))
                
            def get_cluster_pred(preds, np_cluster):
            
                pred_vector = []
                
                for k in range(preds.shape[0]): 
                    temp_sub = np_cluster - preds[k]
                    temp_sub = np.absolute(temp_sub).sum(axis = 1)
                    min_indexes = np.where(temp_sub == temp_sub.min())[0].tolist()
                    # Filter minimum indexes
                    get_min_samples = np_cluster[min_indexes]
                    cosine_sim_score = []
                    for i in range(get_min_samples.shape[0]):
                        cosine_sim_score.append(cal_cosine_sim(get_min_samples[i], preds[k]))
            
                    args_ind = np.argmax(cosine_sim_score)
                    pred_vector.append(get_min_samples[args_ind].reshape(1, -1))
                pred_vector = np.concatenate(pred_vector, axis = 0)
                return pred_vector

            
            for i, prop in enumerate(props):
                error = mae(y_test[:, i], preds[:, i])
                score = r2_score(y_test[:, i], preds[:, i])

                print(f'MAE for {prop}: {error}')
                print(f'R2 Score for {prop}: {score}')

                plt.figure(figsize=(20, 9))
                plt.plot(y_test[:, i])
                plt.plot(preds[:,i])
                plt.savefig(prop+'_new.png')    
                plt.clf()
                metrics['MAE'][prop] = error
                metrics['R2'][prop] = score
            print('NEW Metrics')
            print(metrics)


train_Hist_HSLColor_Feature_Bank()
# train_new()
# train_Histogram_ExternalFeatures()
# train_Histogram_ProminentColor_Histogram_ExternalFeatures()
# train_Histogram_ProminentColor()
# train_Histogram_ProminentColors_ExternalFeatures()
