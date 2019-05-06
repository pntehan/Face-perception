import cv2
import os
from PIL import Image
import numpy as np
import time
from keras.models import load_model

label = {0:'yuhao', 1:'zhouhan'}

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_default.xml")

model = load_model('me.face.model.h5')

cap = cv2.VideoCapture('./video/test.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('./result.mp4', -1, fps, size)

def facePredict(image):
	'''对图片进行缩小并预测'''
	gray = Image.fromarray(image)
	new = gray.resize((100, 100), Image.ANTIALIAS)
	new = np.array(new).reshape(-1, 1, 100, 100)
	result = np.argmax(model.predict(new), axis=1)
	return result[0]

start_time = time.time()
while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	
	count = 1
	for (x, y, w, h) in faces:
		roi_gray = gray[y-10:y+h+10, x-10:x+w+10]
		roi_color = frame[y-10:y+h+10, x-10:x+w+10]

		name = label[facePredict(roi_gray)]
		print(name)
		font = cv2.FONT_HERSHEY_SIMPLEX
		color = (255, 0, 0)
		stroke = 2
		cv2.putText(frame, name, (x+30, y+30), font, 1, color, stroke, cv2.LINE_AA)

		# id_, conf = recognizer.predict(roi_gray)
		# if conf>=45 and conf<=85:
		# 	print(labels[id_])
		# 	font = cv2.FONT_HERSHEY_SIMPLEX
		# 	name = labels[id_]
		# 	color = (255, 0, 0)
		# 	stroke = 2
		# 	cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

		color = (0, 0, 255)
		stroke = 2
		cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, stroke)

		# cv2.imwrite('./image/huxiaoyu/{:.1f}s.jpg'.format(time.time()-start_time), roi_color)

	out.write(frame)
	cv2.imshow('FACES', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

out.release()
cap.release()
cv2.destroyAllWindows()










