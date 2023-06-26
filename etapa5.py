import cv2
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy
 
objeto = cv2.barcode_BarcodeDetector()

def funcion_principal():
	mypath='E:\\Documents\\Juan de Dios\\TFG\\Dataset etapa 5'
	onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
	images = numpy.empty(len(onlyfiles), dtype=object)
	for n in range(0, len(onlyfiles)):
		images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
		images[n] = cv2.cvtColor(images[n], cv2.COLOR_BGR2RGB)
		height, width, channels = images[n].shape 

		ok, decoded_info, decoded_type, corners = objeto.detectAndDecode(images[n])		
		
		# Detección, códigos decodificados, tipos de códigos y esquinas de los códigos:
		print('Código de barra decodificado:', ok)    
		print('Decoded info:',decoded_info)
		print('Decoded type:',decoded_type)		
		print('Corners:', corners)					 #"Corners" desde esquina inferior izquierda en sentido horario

		img_copy = images[n].copy()
		
		#No se ha detectado código de barras
		if ok == False:
			print('Código de barra no detectado')
			print('')

			plt.subplot(111),plt.imshow(images[n])
			plt.title('Original Image'), plt.xticks([]), plt.yticks([])
			plt.show()

		#Se ha detectado código de barras
		else:
			for i in range(len(corners)):  # Para todos los códigos de barras
				lista_x = []
				lista_y = []
				for j in range(4): 							 #Para un código de barras concreto           
					x = int(corners[i][j][0])                #Desde esquina inferior izquierda en sentido horario
					y = int(corners[i][j][1])
					lista_x.append(x)
					lista_y.append(y)
					cv2.circle(img_copy, (int(x),int(y)), 40, (255,0,0), -1) 
				print('Lista x:', lista_x)
				print('Lista y:', lista_y)
				print('')

			plt.subplot(121),plt.imshow(images[n])
			plt.title('Original Image'), plt.xticks([]), plt.yticks([])
			plt.subplot(122),plt.imshow(img_copy)
			plt.title('Barcode detection'), plt.xticks([]), plt.yticks([])
			plt.show()
	
if __name__ == "__main__":              
	funcion_principal()

	
