import numpy as np
np.random.seed(1337)
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Input
from keras.optimizers import Adam
from glob import glob
from PIL import Image

label_male = {0:'male', 1:'female'}
label_smile = {0:'smiling', 1:'not smiling'}
label_glasses = {0:'wearing', 1:'not wearing'}

def load_Image(path):
	'''转换照片格式'''
	img = Image.open(path)
	# half_the_width = img.size[0]/2
	# half_the_height = img.size[1]/2
	# new = img.crop((
	# 	half_the_width-75,
	# 	half_the_height-75,
	# 	half_the_width+75,
	# 	half_the_height+75
	# 	))
	new = img.resize((28, 28), Image.ANTIALIAS)
	return np.array(new)

def get_DataInfo():
	'''返回训练集与测试集的信息'''
	train = []
	with open('./image/MTFL/training.txt', 'r') as fp:
		for line in fp.readlines():
			info = line.strip('\n').split(' ')
			train.append({'img_path':info[1], 'label_male':info[-4], 'label_smile':info[-3], 'label_glasses':info[-2]})
	test = []
	with open('./image/MTFL/testing.txt', 'r') as fp:
		for line in fp.readlines():
			info = line.strip('\n').split(' ')
			test.append({'img_path':info[1], 'label_male':info[-4], 'label_smile':info[-3], 'label_glasses':info[-2]})
	return train, test

def get_DataSet(trainInfo, testInfo):
	'''返回训练集，测试集，训练特征，测试特征'''
	# 0:4151 size is 250*250, 7650: size is 216*160 4151:7650 size is 400*400
	# train_X_1 = np.array([load_Image('./image/MTFL/'+info['img_path']) for info in trainInfo[:4151]])
	# train_X_2 = np.array([load_Image('./image/MTFL/'+info['img_path']) for info in trainInfo[4151:7560]])
	# train_X_3 = np.array([load_Image('./image/MTFL/'+info['img_path']) for info in trainInfo[7560:]])
	train_X = np.array([load_Image('./image/MTFL/'+info['img_path']) for info in trainInfo[:1000]])
	train_y = np.array([int(info['label_male'])-1 for info in trainInfo[:1000]])
	test_X = np.array([load_Image('./image/MTFL/'+info['img_path']) for info in testInfo[:300]])
	test_y = np.array([int(info['label_male'])-1 for info in testInfo[:300]])
	return train_X, train_y, test_X, test_y

def autoencoderModel(dataSet, input_img, encoding_dim=784):
	'''自编码模型构建并返回模型'''
	# encoder layers
	encoded = Dense(11250, activation='relu')(input_img)
	encoded = Dense(5000, activation='relu')(encoded)
	encoded = Dense(1250, activation='relu')(encoded)
	encoder_output = Dense(encoding_dim)(encoded)

	# decoder layers
	decoded = Dense(1250, activation='relu')(encoder_output)
	decoded = Dense(5000, activation='relu')(decoded)
	decoded = Dense(11250, activation='relu')(decoded)
	decoded = Dense(22500, activation='tanh')(decoded)

	# construct the autoencoder model
	autoencoder = Model(input=input_img, output=decoded)

	# construct the encoder model for plotting
	encoder = Model(input=input_img, output=encoder_output)

	# compile autoencoder
	autoencoder.compile(optimizer='adam', loss='mse')

	# training
	autoencoder.fit(dataSet, dataSet, epochs=20, batch_size=256, shuffle=True)

	return autoencoder

def CNN_ModelBuild():
	'''CNN模型构建'''
	model = Sequential() # 创建顺惯模型样本
	# 添加第一层
	model.add(Convolution2D(
		batch_input_shape=(None, 1, 28, 28),
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
	trainInfo, testInfo = get_DataInfo() # 获取数据信息
	X_train, y_train, X_test, y_test = get_DataSet(trainInfo, testInfo) # 读取图片数据和特征数据
	# 图片自编码数据格式准备
	X_train = X_train.reshape((X_train.shape[0], -1))
	X_test = X_test.reshape((X_test.shape[0], -1))
	# # 对数据进行自编码
	# input_img = Input(shape=(22500, ))
	# encoderModel = autoencoderModel(X_train, input_img)
	# X_train = encoderModel.predict(X_train)
	# X_test = encoderModel.predict(X_test)
	# print(X_train.shape, X_test.shape)
	# exit(0)
	# 对输入数据添加高度
	X_train = X_train.reshape(-1, 1, 28, 28)/255
	X_test = X_test.reshape(-1, 1, 28, 28)/255
	# 特征数据格式转化
	y_train = np_utils.to_categorical(y_train, num_classes=2)
	y_test = np_utils.to_categorical(y_test, num_classes=2)
	# 模型构建与训练
	model = CNN_ModelBuild()
	print('训练集数据训练中..................')
	model.fit(X_train, y_train, epochs=5, batch_size=64)
	print('\n测试集数据开始检测................')
	loss, accuracy = model.evaluate(X_test, y_test)

	print('\n模型准确率:', accuracy)

