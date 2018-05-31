#	Author: Victor de A. Amorim		NUSP: 9277642
#	SCC0251 | 2018_1Sem
#	Trabalho 4 - Filtragem 2D

import numpy as np
import imageio
import sys

# Funcao para realizar a filtragem do tipo arbitraria
def f_arb(img):
	N, M = img.shape
	
	# Leitura da dimensao do filtro
	v_input = str(input())
	v_input = v_input.split(" ")
	h = int(v_input[0])
	w = int(v_input[1])

	# Declaracao do filtro no tamanho da imagem para facilitar a convolucao, preenchendo de zero
	filt = np.zeros((N, M))

	# Leitura do filtro
	for i in range(h):
		v_input = str(input())
		filt_input = v_input.split(" ")
		for j in range(w):
			filt[i, j] = float(filt_input[j])

	# Convolucao
	W = np.fft.fft2(img)
	F = np.fft.fft2(filt)
	i_out = np.multiply(W, F)

	return i_out

# Funcao para realizar a laplaciana da gaussiana
def f_laplac_gauss(img):	
	N, M = img.shape

	n = int(input())
	sigma = float(input())
	L = 5
	filt = np.zeros((n, n))

	# Preenchimento do filtro
	for i in range(n):
		for j in range(n):
			s = np.linspace(-5, 5, n);
			x = s[i]
			y = s[n - j - 1]
			filt[i, j] = (-1/(np.pi * np.power(sigma, 4)))*(1 - ((np.power(x, 2) + np.power(y, 2))/(2*np.power(sigma, 2))))*np.exp(-((np.power(x, 2) + np.power(y, 2))/(2*np.power(sigma, 2))))

	# Normalizacao para que a soma dos valores do filtro seja 0
	pos = np.sum(filt[np.where(filt > 0)])
	neg = np.sum(filt[np.where(filt < 0)])
	filt[np.where(filt < 0)] = filt[np.where(filt < 0)] * (-pos/neg)

	# O filtro deve ter o mesmo tamanho que a imagem
	filt_pad = np.zeros((N, M))
	filt_pad[0:n, 0:n] = filt

	# Convolucao
	W = np.fft.fft2(filt_pad)
	F = np.fft.fft2(img)
	i_out = np.multiply(W, F)

	return i_out

# Funcao que realiza a convolução
def f_convolution_pad(img, filt):
	N,M = img.shape
	n,m = filt.shape

	a = int((n-1)/2)
	b = int((m-1)/2)

	# Realiza-se o padding na imagem com zeros
	img_pad = np.pad(img, (a,b), 'constant', constant_values=(0))
	filt_flip = np.flip(np.flip(filt, 0), 1)

	g = np.zeros((N, M))

	# Percorre-se a imagem sem calcular os valores para o padding
	for x in range(a,N-a):
		for y in range(b,M-b):
			# Sub-imagem em que sera realizada o calculo
			sub_img = img_pad[(x-a):(x+a+1), (y-b):(y+b+1)]
			# Preenche-se g desconsiderando o padding
			g[x-a,y-b] = np.sum(np.multiply(sub_img, filt_flip))

	return g

# Funcao para realizar a filtragem do tipo arbitraria
def f_sobel(img):
	# Filtros para realizar a convolucao
	f_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
	f_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

	# Convolucao
	i_x = f_convolution_pad(img, f_x)
	i_y = f_convolution_pad(img, f_y)
	i_out = np.sqrt(np.power(i_x, 2) + np.power(i_y, 2))
	i_out = np.fft.fft2(i_out)

	return i_out

# Funcao para realizar os cortes
def f_cut(img):
	H, W = img.shape

	# Utiliza 1/4 da imagem original
	i_cut1 = img[0:int(H/2), 0:int(W/2)]

	# Leitura dos cortes
	v_input = str(input())
	cortes_input = v_input.split(" ")
	Hlb = float(cortes_input[0])
	Hub = float(cortes_input[1])
	Wlb = float(cortes_input[2])
	Wub = float(cortes_input[3])

	H_cut1, W_cut1 = i_cut1.shape

	# Realiza o corte no 1/4
	i_cut2 = i_cut1[int(Hlb*H_cut1):int(Hub*H_cut1), int(Wlb*W_cut1):int(Wub*W_cut1)]

	return i_cut2

# Funcao para o algoritmo 1NN
def f_1NN(img):
	# Muda a imagem para vetor
	img_v = np.asarray(img).reshape(-1)

	# Dataset e labels
	dataset_fn = str(input()).rstrip()
	labels_fn = str(input()).rstrip()
	dataset = np.load(dataset_fn)
	labels = np.load(labels_fn)

	ex = dataset.shape[0]

	# Guarda-se em um vetor a distancia para cada linha do dataset
	dist = np.zeros(ex)
	for i in range(ex):
		dist[i] = np.linalg.norm(img_v - dataset[i])

	# Posicao no vetor com o menor valor de distancia
	P = np.argmin(dist)

	print(labels[P])
	print(P)

def main():
	# Nomes dos arquivos base
	img_fn = str(input()).rstrip()

	# Imagens a serem utilizadas
	img = imageio.imread(img_fn)
	
	# Opcao de filtragem
	op_filt = int(input())

	# Chamada para a funcao escolhida
	if op_filt == 1: # Arbitrario
		img_out = f_arb(img)
	elif op_filt == 2: # Laplaciana da Gaussiana
		img_out = f_laplac_gauss(img)
	elif op_filt == 3: # Operador Sobel
		img_out = f_sobel(img)
	else:
		print("Entrada invalida!")
		return 1

	#Cortes
	i_cut2 = f_cut(img_out)

	#Classificacao
	f_1NN(i_cut2)

if __name__ == "__main__":
	main()