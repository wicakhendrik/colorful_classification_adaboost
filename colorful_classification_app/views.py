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
    # Skor AdaBoost mentah: sum(alpha_t * h_t(x))
    raw = np.dot(a, y_pred)[0]
    # Hasil Prediksi (+1 / -1 / 0)
    return np.sign(raw), raw

def get_fitur(image_path, number_of_colors):
    image = cv2.imread(os.path.join(settings.BASE_DIR, (image_path)))
    if image is None:
        raise ValueError('File bukan gambar yang valid')
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

    # Palet warna dominan (untuk ditampilkan di UI), diurutkan dari yang terbanyak
    total = float(sum(counts.values()))
    palette = [
        {
            "hex": '#%02x%02x%02x' % tuple(int(round(c)) for c in clf.cluster_centers_[k]),
            "percent": round(counts[k] / total * 100, 1),
        }
        for k in sorted(counts, key=counts.get, reverse=True)
    ]

    return fitur, palette

def image_pro(request):
    if request.method == "POST":
        file_img = request.FILES.get('input_img')
        if file_img is None:
            return render(request, 'template_image_processing.html',
                          {"error": "Pilih sebuah gambar terlebih dahulu."})

        fs = FileSystemStorage()
        filename = fs.save(file_img.name, file_img)
        fnameTest = os.path.join(settings.MEDIA_ROOT, filename)

        try:
            fitur, palette = get_fitur(fnameTest, 20)
        except ValueError:
            fs.delete(filename)
            return render(request, 'template_image_processing.html',
                          {"error": "File tersebut bukan gambar yang valid. Coba PNG atau JPG."})

        fitur = fitur.astype(int).reshape((fitur.shape[0],))
        pr, raw = predict([fitur], a, wl)
        score = float(raw / a.sum())   # -1 (sangat tidak colorful) ... +1 (sangat colorful)

        context = {
            "result": {
                "url_img": settings.MEDIA_URL + filename,
                "filename": file_img.name,
                "is_colorful": bool(pr > 0),
                "label": 'Colorful' if pr > 0 else 'Nocolorful',
                "score": round(score, 2),
                "meter": round((score + 1) / 2 * 100),   # 0..100 untuk penanda di UI
                "palette": palette,
                "top_palette": palette[:6],
            }
        }
        return render(request, 'template_image_processing.html', context)

    return render(request, 'template_image_processing.html')
