#	Author: Victor de A. Amorim		NUSP: 9277642
#	SCC0251 | 2018_1Sem
#	Trabalho 3 - Filtragem 1D

import numpy as np
import imageio
import sys

# Funcao para realizar a normalizacao
def f_norm(img):
	img_max = np.max(img)
	img_min = np.min(img)
	img_norm = (img - img_min) / (img_max - img_min)
	return img_norm

# Funcao para transformar a imagem em um array
def img_to_array(img, N, M):
	img_v = np.zeros(N * M)

	k = 0
	for i in range(N):
		for j in range(M):
			img_v[k] = img[i, j]
			k = k + 1

	return img_v

# Funcao para transformar um array em imagem
def array_to_img(img_v, N, M):
	img = np.zeros((N, M))

	k = 0
	for i in range(N):
		for j in range(M):
			img[i, j] = img_v[k]
			k = k + 1

	return img

# Funcao para dominio espacial
def f_dom_esp(img_v, filt, N, M, n):
	img_out_v = np.zeros(N * M)

	# Percorre todos os valores da imagem
	for i in range(N * M):
		value = 0.0
		k = 0
		if n % 2 != 0: # Filtro de tamanho impar
			for j in range(-int(n/2), 1 + int(n/2)):
				if j < 0:
					value += filt[k] * img_v[(i + j)%(N*M)]
				elif j == 0:
					value += filt[k] * img_v[i]
				else:
					value += filt[k] * img_v[(i + j)%(N*M)]
				k = k + 1
		else: # Filtro de tamanho par
			for j in range(-n/2, n/2):
				if (j == -1 or j == 0):
					value += (filt[k] * img_v[i])/2
				elif j < 0:
					value += filt[k] * img_v[(i + j + 1)%(N*M)]
				else:
					value += filt[k] * img_v[(i + j)%(N*M)]
				k = k + 1
		img_out_v[i] = value

	img_out = array_to_img(img_out_v, N, M)

	return img_out

# Funcao para transformada de Fourier
def transf_fourier_1D(A):

	# Vetor para armazenar a transformada
	F = np.zeros(A.shape, dtype=np.complex64)
	n = A.shape[0]

	# Criar indices para 'x'
	x = np.arange(n)

	for u in np.arange(n):
		# Avaliar a 'similaridade' de cada ponto do sinal com a exponencial complexa na frequencia 'u'
		F[u] = np.sum(np.multiply(A, np.exp((-1j * 2 * np.pi * u * x) / n)))
	return F

# Funcao para transformada inversa de Fourier
def inv_transf_fourier_1D(F):
	# Vetor para armazenar a transformada
	A = np.zeros(F.shape, dtype=np.float32)
	n = F.shape[0]

	# Cria os indices para 'u' (cada frequencia)
	u = np.arange(n)

	for x in np.arange(n):
		# Exponencial complexa relativos a frequencia u
		A[x] = np.real(np.sum(np.multiply(F, np.exp((1j * 2 * np.pi * u * x) / n))))
	
	return A/n

# Funcao para dominio de frequencia
def f_dom_freq(img_v, filt, N, M, n):
	W = transf_fourier_1D(img_v)
	F = transf_fourier_1D(filt)

	img_out_v = inv_transf_fourier_1D(np.multiply(W, F))

	img_out = array_to_img(img_out_v, N, M)

	return img_out


# Funcao para realizar a filtragem do tipo arbitraria
def f_arb(img, n):
	N, M = img.shape
	img_v = img_to_array(img, N, M)
	filt = np.zeros(n)
	v_input = input()
	filt_input = v_input.split(" ")
	for i in range(n):
		filt[i] = float(filt_input[i])

	d = int(input())

	if d == 1: # Dominio Espacial
		img = f_dom_esp(img_v, filt, N, M, n)
	elif d == 2: # Dominio da Frequencia
		filt.resize(N * M)
		img = f_dom_freq(img_v, filt, N, M, n)
	else:
		print("Entrada invalida!")
		return 1

	return img


# Funcao para realizar a filtragem do tipo arbitraria
def f_gauss(img, n):	
	N, M = img.shape
	img_v = img_to_array(img, N, M)
	
	sigma = float(input())
	filt = np.zeros(n)

	j = 0
	if n % 2 != 0:
		for i in range(-int(n/2), 1 + int(n/2)):
			filt[j] = (1/(np.sqrt(2*np.pi)))*np.exp((-(i)**2)/(2*(sigma)**2))
			j = j + 1
	else:
		for i in range(-n/2, n/2):
			filt[j] = (1/(np.sqrt(2*np.pi)))*np.exp((-(i)**2)/(2*(sigma)**2))
		j = j + 1

	# Normalizacao
	filt = filt / np.sum(filt)

	d = int(input())

	if d == 1: # Dominio Espacial
		img = f_dom_esp(img_v, filt, N, M, n)
	elif d == 2: # Dominio da Frequencia
		filt.resize(N * M)
		img = f_dom_freq(img_v, filt, N, M, n)
	else:
		print("Entrada invalida!")
		return 1

	return img

# Funcao para o calculo do erro
def RMSE(img, img_out):
	# Dimensao obtida a partir da H
	N, M = img.shape

	# Calculo do erro
	count = 0
	for i in range(N):
		for j in range(M):
			count += (int(img[i, j]) - int(img_out[i, j])) ** 2
			
	return np.sqrt(count/(N*M))

def main():
	# Nomes dos arquivos base
	img_fn = str(input()).rstrip()

	# Imagens a serem utilizadas
	img = imageio.imread(img_fn)
	
	# Opcao de filtragem
	op_filt = int(input())
	n = int(input())

	# Chamada para a funcao escolhida
	if op_filt == 1: # Arbitrario
		img_out = f_arb(img, n)
	elif op_filt == 2: # Funcao Gaussiana
		img_out = f_gauss(img, n)
	else:
		print("Entrada invalida!")
		return 1

	img_out_norm = f_norm(img_out)
	img_out_norm = (img_out_norm * 255).astype(np.uint8)

	# Comparacao
	erro = RMSE(img, img_out_norm)
	print(np.round(erro, 4))


if __name__ == "__main__":
	main()