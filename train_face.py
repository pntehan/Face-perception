import numpy as np
np.random.seed(1337)
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from glob import glob
from PIL import Image

label = {'yuhao':0, 'zhouhan':1}

def load_Image(path):
	'''转换照片格式'''
	img = Image.open(path).convert('L')
	new = img.resize((100, 100), Image.ANTIALIAS)
	return np.array(new)

def get_DataSet(dirs):
	'''返回训练集，测试集，训练特征，测试特征'''
	# 0:4151 size is 250*250, 7650: size is 216*160 4151:7650 size is 400*400
	train_X = np.array([load_Image(path) for path in dirs])
	train_y = np.array([label[x.split('\\')[0].split('/')[-1]] for x in dirs])
	# test_X = np.array([load_Image(path) for path in dirs])
	# test_y = np.array([label[x.split('\\')[0].split('/')[-1]] for x in dirs])
	return train_X, train_y

def get_TestDataSet(dirs):
	'''返回训练集，测试集，训练特征，测试特征'''
	# 0:4151 size is 250*250, 7650: size is 216*160 4151:7650 size is 400*400
	train_X = np.array([load_Image(path) for path in dirs])
	train_y = np.array([label[x.split('\\')[1].split('/')[-1]] for x in dirs])
	# test_X = np.array([load_Image(path) for path in dirs])
	# test_y = np.array([label[x.split('\\')[0].split('/')[-1]] for x in dirs])
	return train_X, train_y

def CNN_ModelBuild():
	'''CNN模型构建'''
	model = Sequential() # 创建顺惯模型样本
	# 添加第一层
	model.add(Convolution2D(
		batch_input_shape=(None, 1, 100, 100),
		filters=32,
		kernel_size=5,
		strides=1,
		padding='same',
		data_format='channels_first',
	))
	model.add(Activation('relu'))
	# 添加池
	model.add(MaxPooling2D(
		pool_size=2,
		strides=2,
		padding='same',
		data_format='channels_first',
	))

	# 添加第二层
	model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('relu'))
	model.add(Dense(2))
	model.add(Activation('softmax'))

	adam = Adam(lr=1e-4)

	model.compile(optimizer=adam,
		loss='categorical_crossentropy',
		metrics=['accuracy'])
	return model


if __name__ == '__main__':
	'''开始执行'''
	dirs1 = glob('./image/DIY/zhouhan/*.jpg')
	dirs2 = glob('./image/DIY/yuhao/*.jpg')
	# dirs3 = glob('./image/DIY/others/*/*.jpg')
	X_train_1, y_train_1 = get_DataSet(dirs1) # 读取图片数据和特征数据
	X_train_2, y_train_2 = get_DataSet(dirs2) # 读取图片数据和特征数据
	X_test, y_test = get_TestDataSet(dirs3)
	# 拼接数据
	X_train = np.vstack((X_train_1, X_train_2))
	# X_test = np.vstack((X_test_1, X_test_2))
	y_train = np.hstack((y_train_1, y_train_2))
	# y_test = np.hstack((y_test_1, y_test_2))
	# print(X_train.shape, X_test.shape)
	# exit(0)
	# 对输入数据添加高度
	X_train = X_train.reshape(-1, 1, 100, 100)/255
	X_test = X_test.reshape(-1, 1, 100, 100)/255
	# 特征数据格式转化
	y_train = np_utils.to_categorical(y_train, num_classes=2)
	y_test = np_utils.to_categorical(y_test, num_classes=2)
	# 模型构建与训练
	model = CNN_ModelBuild()
	print('训练集数据训练中..................')
	model.fit(X_train, y_train, epochs=1, batch_size=64)
	print('\n测试集数据开始检测................')
	loss, accuracy = model.evaluate(X_test, y_test)

	print('\n模型准确率:', accuracy)
	model.save('./me.face.model.h5')
