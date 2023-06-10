import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# 載入 SavedModel
model_path = 'Model_M' #載入對應欲推論結果之Model
loaded_model = tf.keras.models.load_model(model_path)


# 狗種類別的標籤，須注意順序，以免於顯示結果miss match
labels = ['Maltese_dog','golden_retriever','Labrador_retriever','collie','Border_collie','malamute','Siberian_husky','Samoyed']

# 預處理影像
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    # image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# 從本機資料夾辨識影像
def predict_from_folder(folder_path):
    images = []
    original_imgs = [] #先建立一個保存原來影像輸入來源的list，以便最後於結果顯示原圖
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path)
        original_imgs.append(image)
        image = preprocess_image(image)
        images.append((file_name, image))
    
    # 預測並顯示結果
    for i, (file_name, image) in enumerate(images):
        prediction = loaded_model.predict(image)
        label_index = np.argmax(prediction)
        label = labels[label_index]
        confidence = prediction[0][label_index]
        plt.imshow(cv2.cvtColor(original_imgs[i], cv2.COLOR_BGR2RGB))  #輸出結果搭配原始輸入影像顯示
        plt.title(f'Image {i+1}: {label} (Confidence: {confidence})')
        plt.axis('off')
        plt.show()
        print(f'Image {file_name}: {label} (Confidence: {confidence})')

# 資料夾路徑設定
folder_path = 'Input'  # 資料夾路徑
predict_from_folder(folder_path)
