import cv2
from glob import glob
from PIL import Image
import numpy as np
import pickle

# 初始化人脸识别器和人脸检测器
face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_label = []

dirs = glob('./image/*/*.jpg')

for path in dirs:
	pil_image = Image.open(path).convert("L")
	image_array = np.array(pil_image, 'uint8')
	label = path.split('\\')[1]

	if not label in label_ids:
		current_id += 1
		label_ids[label] = current_id
	id_ = current_id

	faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

	for (x, y, w, h) in faces:
		roi = image_array[y:y+h, x:x+w]
		x_train.append(roi)
		y_label.append(id_)

with open('label.pickle', 'wb') as f:
	pickle.dump(label_ids, f)

print(label_ids)
print(y_label)
recognizer.train(x_train, np.array(y_label))
recognizer.save('trainner.yml')












