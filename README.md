# Product-recommender-system

Dataset -  https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

A simple image-based product recommender built with ResNet50 and Nearest Neighbors. Upload an image and the app returns visually similar items (it searches a gallery of ~44,000 images using precomputed embeddings).

### **Project summary:**

When a user uploads an image, the app:

1. Extracts a feature vector (embedding) from the image using a pretrained ResNet50 (ImageNet weights) with a global max pooling head.


2. Normalizes the embedding.


3. Searches the embedding index (44k image embeddings) using sklearn.neighbors.NearestNeighbors (Euclidean distance) to find the closest images.

4. Displays the top-N visually similar items in the Streamlit UI.

This is implemented as a lightweight Streamlit app so you can demo recommendations in seconds.
