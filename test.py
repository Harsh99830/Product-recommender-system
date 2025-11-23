import pickle
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import tensorflow
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

feature_list = np.array(pickle.load(open('feature_list.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))
model=ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([                  #changing the 7x7 into 1x1
    model,
    GlobalMaxPooling2D(),
])

img = image.load_img('sample/jersey.png', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)  # changing 3d into 4d(batch_size, height, width, channels)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=5, metric='euclidean',algorithm='brute')
neighbors.fit(feature_list)

distances,indices = neighbors.kneighbors([normalized_result])
print(indices)

for file in indices[0]:
    print(filenames[file])