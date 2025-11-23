import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([                  #changing the 7x7 into 1x1
    model,
    GlobalMaxPooling2D(),
])

# print(model.summary())

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)      # changing 3d into 4d(batch_size, height, width, channels)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

filenames = []
for file in os.listdir("./dataset/images"):
    filenames.append(os.path.join("./dataset/images",file))

feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list,open("feature_list.pkl","wb"))
pickle.dump(filenames,open("filenames.pkl","wb"))
