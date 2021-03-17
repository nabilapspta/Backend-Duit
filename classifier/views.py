from rest_framework.views import APIView
from rest_framework.parsers import FileUploadParser
from .helper.response import Response
from rest_framework import status
import cv2
import numpy as np
import pickle
from PIL import Image
from scipy.stats import skew,entropy, kurtosis
import os
from sklearn import svm
from sklearn.datasets import make_classification
from skimage import measure
from skimage.feature import greycomatrix, greycoprops
# Create your views here.

def histogram(image):
    img = image.astype(int)
    sk = np.array ([x for x in range (256)], dtype= int)
    tinggi, lebar = img.shape
    n = tinggi*lebar
    nk = np.array([ np.count_nonzero(img == i) for i in range (256) ], dtype = int)
    p = np.array (nk/n)
    return p

def rataE(e):
    temp=[]
    for i in range(len(e)):
        x=i*e[i]
        temp.append(x)
    return sum(temp)

class Data:
    def __init__(self, path):
        img=np.array(Image.open(path))
        width = 512
        height = 256
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        self.hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        img_gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        self.citra = img_gray
        

class ClassifierView(APIView):
	parser_class = (FileUploadParser, )

	def post(self, request):
		file = request.data.get('file', None)

		if not file:
			print(sklearn.__version__)
			return Response.badRequest(
				data = None,
				status = status.HTTP_400_BAD_REQUEST,
				message = 'Bad Req'
			)

		#get data
		data_citra = Data(file)

		#tahap 1 cek keaslian


		color_feature = []
		#extract HSV value
		h = histogram(data_citra.hsv[0])
		s = histogram(data_citra.hsv[1])
		v = histogram(data_citra.hsv[2])

		#H Feature from HSV
		sh = skew(h)
		dh = np.std(h)
		ratah= rataE(h)
		enh= entropy(h)
		kurh= kurtosis(h)

		#S Feature from HSV
		ss = skew(s)
		ds = np.std(s)
		ratas= rataE(s)
		ens= entropy(s)
		kurs= kurtosis(s)

		#V Feature from HSV
		sv = skew(v)
		dv = np.std(v)
		ratav= rataE(v)
		env= entropy(v)
		kurv= kurtosis(v)

		color_feature.append(sh)
		color_feature.append(dh)
		color_feature.append(ratah)
		color_feature.append(enh)
		color_feature.append(kurh)

		color_feature.append(ss)
		color_feature.append(ds)
		color_feature.append(ratas)
		color_feature.append(ens)
		color_feature.append(kurs)

		color_feature.append(sv)
		color_feature.append(dv)
		color_feature.append(ratav)
		color_feature.append(env)
		color_feature.append(kurv)


		color_feature = np.reshape(color_feature, (1, -1))


		#import model for cek keaslian
		work_dir = os.getcwd()
		keaslian_model = pickle.load(open(work_dir+'/classifier/classifier_file/keaslian_model.sav', 'rb'))

		cek_keaslian = keaslian_model.predict(color_feature)
		data = {
			'cek_keaslian' : cek_keaslian.tolist()[0],
			'nominal' : None,
			'kelayakan': None
		}
		if cek_keaslian == False:
			return Response.ok(
				data = data,
				status = status.HTTP_200_OK,
				message = 'Hasil klasifikasi'
			)

		#step 2, cek nominal with GLCM

		
		glcm_feature = []
		glcm = greycomatrix(data_citra.citra, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],  symmetric = True, normed = True )
		contrast = greycoprops(glcm, 'contrast')
		homogeneity = greycoprops(glcm, 'homogeneity')
		energy = greycoprops(glcm, 'energy')
		correlation = greycoprops(glcm, 'correlation')
		asm = greycoprops(glcm, 'ASM')

		glcm_feature.append(contrast[0][0])
		glcm_feature.append(contrast[0][1])
		glcm_feature.append(contrast[0][2])
		glcm_feature.append(contrast[0][3])

		glcm_feature.append(homogeneity[0][0])
		glcm_feature.append(homogeneity[0][1])
		glcm_feature.append(homogeneity[0][2])
		glcm_feature.append(homogeneity[0][3])

		glcm_feature.append(energy[0][0])
		glcm_feature.append(energy[0][1])
		glcm_feature.append(energy[0][2])
		glcm_feature.append(energy[0][3])


		glcm_feature.append(correlation[0][0])
		glcm_feature.append(correlation[0][1])
		glcm_feature.append(correlation[0][2])
		glcm_feature.append(correlation[0][3])

		glcm_feature.append(asm[0][0])
		glcm_feature.append(asm[0][1])
		glcm_feature.append(asm[0][2])
		glcm_feature.append(asm[0][3])

		glcm_feature = np.reshape(glcm_feature, (1, -1))

		#import model for cek nominal
		work_dir = os.getcwd()
		nominal_model = pickle.load(open(work_dir+'/classifier/classifier_file/nominal_model.sav', 'rb'))

		nominal = nominal_model.predict(glcm_feature)
		nominal = nominal.tolist()[0]
		data.update(nominal=nominal)

		print(data)
		kelayakan=False

		if(nominal==1):
		    edges = cv2.Canny(data_citra.citra,10,10)
		    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		    cnt=[]
		    imgCanny=data_citra.citra.copy()
		    if len(contours) != 0:
		        c = max(contours, key = cv2.contourArea)
		        cnt.append(c)
		        cv2.drawContours(imgCanny, contours, -1, (0,255,0), 3)
		    luas = cv2.contourArea(cnt[0])
		    print("Luas: "+str(luas)+" px")

		    if(luas>4271.5):
		        kelayakan=True
		    #seribu
		elif(nominal==2):
		    edges = cv2.Canny(data_citra.citra,10,10)
		    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		    cnt=[]
		    imgCanny=data_citra.citra.copy()
		    if len(contours) != 0:
		        c = max(contours, key = cv2.contourArea)
		        cnt.append(c)
		        cv2.drawContours(imgCanny, contours, -1, (0,255,0), 3)
		    luas = cv2.contourArea(cnt[0])
		    print("Luas: "+str(luas)+" px")
		    if(luas>4307.5):
		    	kelayakan=True
		    #2ribu
		elif(nominal==5):
		    edges = cv2.Canny(data_citra.citra,10,10)
		    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		    cnt=[]
		    imgCanny=data_citra.citra.copy()
		    if len(contours) != 0:
		        c = max(contours, key = cv2.contourArea)
		        cnt.append(c)
		        cv2.drawContours(imgCanny, contours, -1, (0,255,0), 3)
		    luas = cv2.contourArea(cnt[0])
		    print("Luas: "+str(luas)+" px")
		    if(luas>3652.0):
		    	kelayakan=True
		    #5ribu
		elif(nominal==10):
		    edges = cv2.Canny(data_citra.citra,10,10)
		    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		    cnt=[]
		    imgCanny=data_citra.citra.copy()
		    if len(contours) != 0:
		        c = max(contours, key = cv2.contourArea)
		        cnt.append(c)
		        cv2.drawContours(imgCanny, contours, -1, (0,255,0), 3)
		    luas = cv2.contourArea(cnt[0])
		    print("Luas: "+str(luas)+" px")
		    if(luas>3518.5):
		    	kelayakan=True
		    #10ribu:
		elif(nominal==20):
		    edges = cv2.Canny(data_citra.citra,10,10)
		    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		    cnt=[]
		    imgCanny=data_citra.citra.copy()
		    if len(contours) != 0:
		        c = max(contours, key = cv2.contourArea)
		        cnt.append(c)
		        cv2.drawContours(imgCanny, contours, -1, (0,255,0), 3)
		    luas = cv2.contourArea(cnt[0])
		    print("Luas: "+str(luas)+" px")
		    if(luas>4376.5):
		    	kelayakan=True
		    #20ribu
		elif(nominal==50):
		    edges = cv2.Canny(data_citra.citra,10,10)
		    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		    cnt=[]
		    imgCanny=data_citra.citra.copy()
		    if len(contours) != 0:
		        c = max(contours, key = cv2.contourArea)
		        cnt.append(c)
		        cv2.drawContours(imgCanny, contours, -1, (0,255,0), 3)
		    luas = cv2.contourArea(cnt[0])
		    print("Luas: "+str(luas)+" px")
		    if(luas>5904.5):
		    	kelayakan=True
		    #50ribu
		else:
		    edges = cv2.Canny(data_citra.citra,10,10)
		    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		    cnt=[]
		    imgCanny=data_citra.citra.copy()
		    if len(contours) != 0:
		        c = max(contours, key = cv2.contourArea)
		        cnt.append(c)
		        cv2.drawContours(imgCanny, contours, -1, (0,255,0), 3)
		    luas = cv2.contourArea(cnt[0])
		    print("Luas: "+str(luas)+" px")
		    if(luas>6770.5):
		    	kelayakan=True
		    #100ribu

		data.update(kelayakan=kelayakan)
		return Response.ok(
			data = data,
			status = status.HTTP_200_OK,
			message = 'Hasil klasifikasi'
		)