import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16

class Similarity:
    def __init__(self, pic: str, data_list):
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = VGG16(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        self.pic = pic
        self.data_list = data_list

    def extract(self, img):
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
    
    
    def save_feature(self):
        features = []
        img_paths = []

        # Iterate through images (Change the path based on your image location)
        for i in range(len(self.data_list)):
            image_path = "./images/" + self.data_list[i]
            img_paths.append(image_path)
            print(self.data_list[i])

            # Extract Features
            feature = self.extract(img=Image.open(image_path))

            features.append(feature)

            # Save the Numpy array (.npy) on designated path
            # feature_path = "./features/" + img_path[:-4] + ".npy"
            # np.save(feature_path, feature)

        return features, img_paths
    
    def target_feature(self):
        # Insert the image query
        img = Image.open("./images/" + self.pic)

        # Extract its features
        query = self.extract(img)
        
        return query


    def calculate_dist(self):
        features, img_paths = self.save_feature()
        # Calculate the similarity (distance) between images
        dists = np.linalg.norm(features - self.target_feature(), axis=1)

        # Extract 30 images that have lowest distance
        ids = np.argsort(dists)[:30]

        scores = [(dists[id], img_paths[id]) for id in ids]
        return scores
    

    def calculate_cos(self):
        features, img_paths = self.save_feature()
        query = self.target_feature()
        # Calculate the similarity (cosine) between images
        cos_sim = []
        for i in range(len(features)):
            cos_sim.append(dot(features[i], query) / (norm(features[i]) * norm(query)))
        # Extract 30 images that have highest distance
        ids = np.argsort(cos_sim)[::-1]
        scores = [(cos_sim[id], img_paths[id]) for id in ids]
        return scores

    
    def visualize(self):
        scores = self.calculate_cos()
        # Visualize the result
        axes=[]
        fig=plt.figure(figsize=(8,8))
        for a in range(5*6):
            score = scores[a]
            axes.append(fig.add_subplot(5, 6, a+1))
            subplot_title=str(round(score[0],2)) + str(score[1][9:])
            axes[-1].set_title(subplot_title)  
            plt.axis('off')
            plt.imshow(Image.open(score[1]))
        fig.tight_layout()
        plt.show()

    
    def cos_similarity(self, cos_list):
        features = self.save_feature()[0]
        index0 = self.data_list.index(cos_list[0])
        index1 = self.data_list.index(cos_list[1])

        cos_sim = dot(features[index0], features[index1]) / (norm(features[index0]) * norm(features[index1]))
        
        print(f'\n{cos_list[0]}와 {cos_list[1]}의 COS Similarity == {cos_sim} 입니다.')