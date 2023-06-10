import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

def DataGenerator(file_path, batch_size):
    
    ImageData = ImageDataGenerator()
    
    data = ImageData.flow_from_directory(
        file_path,
        target_size=(224, 224),
        color_mode='rgb',
        classes={
                 'Maltese_dog': 0,
                 'golden_retriever': 1,
                 'Labrador_retriever': 2,
                 'collie': 3,
                 'Border_collie': 4,
                 'malamute': 5,
                 'Siberian_husky': 6,
                 'Samoyed': 7,
                 },
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
    )
    
    return data

test_dataset = DataGenerator(file_path = 'Test', batch_size=8)

for model_name in ["Model_S", "Model_M", "Model_H"]:

    model_path = model_name
    model = tf.keras.models.load_model(model_path)

    t1 = time.time()
    test_result = model.evaluate(test_dataset)
    t2 = time.time()

    print(f"{model_name} Accuracy(original): {test_result[1]:.2%}") #1是accuracy, 0是loss
    print(f"{model_name} Time(original): {t2-t1}")

    def Dataset2Numpy(dataset):
        
        y = []
        x = []

        for i in range(len(dataset)):
            x.append(dataset[i][0]) 
            y.append(dataset[i][1])
            
        x = np.concatenate((x),axis=0)
        y = np.concatenate((y),axis=0)
        x = np.expand_dims(x, axis=1)
        
        return x, y

    x, y = Dataset2Numpy(test_dataset)


    # 載入 SavedModel (TensorflowLit格式)

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=f"{model_path}.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_pred = []
    y_true = []

    # Test the model on random input data.
    input_shape = input_details[0]['shape']

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.

    t1 = time.time()

    for i in range(y.shape[0]):

        interpreter.set_tensor(input_details[0]['index'], x[i])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred = np.argmax(output_data)
        y_pred.append(pred)
        true = np.argmax(y[i])
        y_true.append(true)

    t2 = time.time()

    acc = accuracy_score(y_true, y_pred)
    print(f"{model_name} Accuracy(Lite): {acc:.2%}") #1是accuracy, 0是lo
    print(f"{model_name} Time(Lite): {t2-t1}")
