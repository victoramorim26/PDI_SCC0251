#	Author: Victor de A. Amorim		NUSP: 9277642
#	SCC0251 | 2018_1Sem
#	Trabalho 5 -  Inpainting

import numpy as np
import imageio
import sys

# Funcao para realizar a normalizacao de uma imagem
def f_norm(img):
	img_max = np.max(img)
	img_min = np.min(img)
	img_norm = (img - img_min) / (img_max - img_min)
	return img_norm

# Funcao para o calculo do erro
def RMSE(img, img_out):
	N, M = img.shape
	return np.sqrt(np.sum(np.power(img - img_out, 2)/(N*M)))

# Funcao para a execucao do algoritmo de Gerchberg Papoulis
def f_gerchberg_papoulis(img_i, img_m, T):
	N, M = img_i.shape

	# Para realizar a convolucao, o filtro deve ter o tamanho da imagem
	filt = np.zeros((N, M))
	
	# Preenche o filtro com a media
	for i in range(7):
		for j in range(7):
			filt[i, j] = 1/49

	# Transformada do Filtro para realizar a convolucao
	W = np.fft.fft2(filt)

	# 1. g0 = g
	G = img_i

	# 2. M = FFT(m)
	MF = np.fft.fft2(img_m)
	MF_max = np.max(MF)

	# 3.
	for k in range(T):
		# a. Gk = FFT(gk-1)
		G = np.fft.fft2(G)

		# b. i. e ii.
		# Encontra as posicoes em G a serem zeradas
		G_max = np.max(G)
		G_v = np.asarray(G).reshape(-1)
		G_v[np.intersect1d(np.where(G_v >= 0.9*MF_max), np.where(G_v <= 0.01*G_max))] = 0
		G = np.reshape(G_v, (N, M))
	
		# d. Convolucao
		G = np.real(np.fft.ifft2(np.multiply(W, G)))

		# e. Normalizacao
		G = f_norm(G)
		G = (G * 255).astype(np.uint8)

		# f. Com a máscara binária, verifica quais pontos serao utilizados de img_i ou de G
		G = (np.multiply((1 - (img_m/255)), img_i)) + (np.multiply((img_m/255), G))

	return G

def main():
	# Nomes dos arquivos base
	imgo_fn = str(input()).rstrip()
	imgi_fn = str(input()).rstrip()
	imgm_fn = str(input()).rstrip()

	# Imagens a serem utilizadas
	img_o = imageio.imread(imgo_fn)
	img_i = imageio.imread(imgi_fn)
	img_m = imageio.imread(imgm_fn)
	
	# Numero de iteracoes
	T = int(input())

	img_i = f_gerchberg_papoulis(img_i, img_m, T)

	# 2)
	img_o = np.uint8(img_o)
	img_i = np.uint8(img_i)

	# Comparacao
	erro = RMSE(img_o, img_i)
	print(np.round(erro, 5))

if __name__ == "__main__":
	main()