#	Author: Victor de A. Amorim		NUSP: 9277642
#	SCC0251 | 2018_1Sem
#	Trabalho 6 -  Restauracao de Imagens

import numpy as np
import imageio
import sys

# Funcao para o calculo do erro
def RMSE(img, img_out):
	N, M = img.shape
	return np.sqrt(np.sum(np.power(img - img_out, 2)/(N * M)))

# Funcao para restauracao usando o filtro adaptativo de reducao de ruido local
def f_local(img_noisy, N):
	# Valor da distribuicao de ruido
	sigma = float(input())
	X, Y = img_noisy.shape

	# Imagem de retorno
	img_out = np.zeros(img_noisy.shape)
	
	# Uso do padding para expandir os limites da matriz, utilizando o mode='wrap' para tornar a matriz circular
	a = b = int((N-1)/2)
	img_pad = np.pad(img_noisy, (a,b), mode='wrap')
	# Filtro 4D para armazenar o filtro para cada posicao da imagem
	filt = np.zeros((X, Y, N, N))

	# Preenchimento do Filtro
	for i in range(a, X + a):
		for j in range(b, Y + b):
			filt[i - a, j - b] = img_pad[(i - a):(i + a + 1), (j - b):(j + b + 1)]

	# Aplicacao do filtro na img_noisy, para gerar a img_out
	img_out = img_noisy - ((np.power(sigma, 2)/np.var(filt, axis = (2, 3))) * (img_noisy - np.mean(filt, axis = (2, 3))))
	return img_out

# Funcao para realizar a primeira etapa do filtro adaptativo de mediana
def f_etapa_a(N, M, img_noisy, filt, i, j):
	z_min = np.min(filt)
	z_max = np.max(filt)
	z_med = np.median(filt)
	A1 = z_med - z_min
	A2 = z_med - z_max

	if A1 > 0 and A2 < 0:
		return f_etapa_b(N, M, img_noisy, filt, i, j)
	else:
		N = N + 1
		if N <= M:
			filt = np.pad(filt, (1, 1), 'edge')
			return f_etapa_a(N, M, img_noisy, filt, i, j)
		else:
			return z_med

# Funcao para realizar a segunda etapa do filtro adaptativo de mediana
def f_etapa_b(N, M, img_noisy, filt, i, j):
	z_min = np.min(filt)
	z_max = np.max(filt)
	z_med = np.median(filt)
	B1 = img_noisy[i, j] - z_min
	B2 = z_med - z_max

	if B1 > 0 and B2 < 0:
		return img_noisy[i, j]
	else:
		return z_med

# Funcao para realizar a restauracao usando o filtro adaptativo de mediana
def f_mediana(img_noisy, N):
	M = int(input())
	X, Y = img_noisy.shape

	# Imagem de retorno
	img_out = np.zeros(img_noisy.shape)
	
	# Uso do padding para expandir os limites da matriz, utilizando o mode='wrap' para tornar a matriz circular
	a = b = int((N-1)/2)
	img_pad = np.pad(img_noisy, (a,b), mode='wrap')
	filt = np.zeros((N, N))

	# Preenchimento do filtro juntamente com a imagem de retorno
	for i in range(a, X + a):
		for j in range(b, Y + b):
			filt = img_pad[(i - a):(i + a + 1), (j - b):(j + b + 1)]
			img_out[i - a, j - b] = f_etapa_a(N, M, img_noisy, filt, i - a, j - b)

	return img_out

# Funcao para realizar a restauracao usando o filtro da média contra-harmônica
def f_media_contra_harmonica(img_noisy, N):
	Q = float(input())
	X, Y = img_noisy.shape

	# Imagem de retorno
	img_out = np.zeros(img_noisy.shape)

	# Uso do padding para expandir os limites da matriz, utilizando o mode='constante' e preenchendo com zero
	a = b = int((N-1)/2)
	img_pad = np.pad(img_noisy, (a,b), mode='constant', constant_values=(0))

	# Preenchimento do filtro juntamente com a imagem de retorno
	for i in range(a, X + a):
		for j in range(b, Y + b):
			sub_img = img_pad[(i - a):(i + a + 1), (j - b):(j + b + 1)]
			sum_den = np.sum(np.power(sub_img[np.where(sub_img != 0)], Q))
			if sum_den != 0:
				img_out[i - a, j - b] = np.sum(np.power(sub_img, Q + 1))/sum_den

	return img_out

def main():
	# Nomes dos arquivos base
	imgcomp_fn = str(input()).rstrip()
	imgnoisy_fn = str(input()).rstrip()

	# Imagens a serem utilizadas
	img_comp = imageio.imread(imgcomp_fn)
	img_noisy = imageio.imread(imgnoisy_fn)
	
	# Opcao de filtragem
	op_filt = int(input())
	N = int(input())

	# Chamada para a funcao escolhida
	if op_filt == 1: # Ruido Local
		img_out = f_local(img_noisy, N)
	elif op_filt == 2: # Mediana
		img_out = f_mediana(img_noisy, N)
	elif op_filt == 3: # Media contra-harmonica
		img_out = f_media_contra_harmonica(img_noisy, N)
	else:
		print("Entrada invalida!")
		return 1

	img_out = np.uint8(img_out)

	# Comparacao
	erro = RMSE(img_comp, img_out)
	print(np.round(erro, 5))

if __name__ == "__main__":
	main()