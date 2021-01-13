from django.shortcuts import render

# Create your views here.
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.conf import settings

import csv
import joblib
import cv2
import os
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

T=10

# Load Bobot Predictor
a_file = csv.reader(open('static/bbt_pred.csv'), delimiter=',')
a = np.zeros(shape=T)
for row in a_file:
    for i, ai in enumerate(row[:-1]):
        a[i] = ai

# Load Model Predictor
wl = np.zeros(shape=T, dtype=object)

for i in range(T):
    urlModel = os.path.join(settings.BASE_DIR, ("static/ds_tree_%s.pkl"%i))
    wl[i] = joblib.load(open(urlModel, 'rb'))

def predict(fitur, a, weak_learn):
    # Mendapatkan hipotesis dari weak learn
    y_pred = np.array([wl.predict(fitur) for wl in weak_learn], dtype=int)
    # Hasil Prediksi
    return np.sign(np.dot(a, y_pred))[0]

def get_fitur(image_path, number_of_colors):
    image = cv2.imread(os.path.join(settings.BASE_DIR, (image_path)))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    
    #convert fitur to np array
    fitur = np.zeros((20,1))
    for i in range(len(counts)):
        fitur[i] = counts[i]
        
    return fitur

def image_pro(request):
    if request.method == "POST":
        filepath = request.FILES.get('input_img', False)
        if filepath != False:
            file_img = request.FILES['input_img']

            fs = FileSystemStorage()
            filename = fs.save(file_img.name, file_img)

            fnameTest = os.path.join(settings.BASE_DIR, ('media/'+file_img.name))

            fitur = get_fitur(fnameTest, 20).astype(int)
            fitur = fitur.reshape((fitur.shape[0],))
            pr = predict([fitur], a, wl)

            context = {
                "output" : [
                    {
                        "url_img" : "/media/"+file_img.name,
                        "identifikasi" : 'Colorful' if pr > 0 else 'Nocolorful',
                    }
                ]
            }

            return render(request, 'template_image_processing.html', context)

    return render(request, 'template_image_processing.html')
