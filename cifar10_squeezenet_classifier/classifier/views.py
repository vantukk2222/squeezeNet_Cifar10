import os
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_json_path = os.path.join(BASE_DIR, 'classifier', 'SqueezeNet_vs_CIFAR10', 'models', 'squeeze_net.json')
model_weights_path = os.path.join(BASE_DIR, 'classifier', 'SqueezeNet_vs_CIFAR10', 'models', 'squeeze_net.weights.h5')

with open(model_json_path, 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)

model.load_weights(model_weights_path)

def index(request):
    if request.method == 'POST' and 'file' in request.FILES:
        img = request.FILES['file']
        img_name = default_storage.save('image_detection/' + img.name, ContentFile(img.read()))
        print("IMAGE PATH: ", img_name)
        img_path = default_storage.path(img_name)

        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0 

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        result = labels[predicted_class]

        return JsonResponse({'result': result})

    return render(request, 'index.html')
