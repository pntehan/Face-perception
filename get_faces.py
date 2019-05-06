import cv2
import sys
from PIL import Image

def catchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
	'''从视频中捕捉人脸图像'''
	cv2.namedWindow(window_name)
	cap = cv2.VideoCapture(camera_idx)
	classifier = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_default.xml")
	color = (0, 255, 0)
	# 按帧读取视频
	num = 1
	while cap.isOpened():
		ok, frame = cap.read()
		if not ok:
			break
		# 将图片转化为灰度图像
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# 人脸检测
		faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
		if len(faceRects) > 0:
			for faceRect in faceRects:
				x, y, w, h = faceRect
				# 保存图片
				img_name = 'image/DIY/%s/%d.jpg'%(path_name, num)
				image = frame[y-10:y+h+10, x-10:x+w+10]
				cv2.imwrite(img_name, image)

				num += 1
				if num > catch_pic_num:
					break
				# 画出矩形框
				cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, 2)
				# 显示当前捕捉图片数量
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(frame, 'num:%d'%(num), (x+30, y+30), font, 1, (255, 0, 255), 4)
		if num > catch_pic_num: break
		# 显示图像
		cv2.imshow(window_name, frame)
		c = cv2.waitKey(10)
		if c & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	# # 录制视频截取图片
	# catchPICFromVideo('cutFaces', int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
	# 导入视频截取图片
	catchPICFromVideo('cutFaces', sys.argv[1], int(sys.argv[2]), sys.argv[3])







