#	Author: Victor de A. Amorim		NUSP: 9277642
#	SCC0251 | 2018_1Sem
#	Trabalho 1 - Gerador de Imagens

import numpy as np
import imageio
import sys
import random

# Funcao 1 para preenchimento da imagem cena
def f1(imgc, C):
	for x in range(C):
		for y in range(C):
			imgc[x, y] = x + y
	return imgc

# Funcao 2 para preenchimento da imagem cena
def f2(imgc, C, Q):
	for x in range(C):
		for y in range(C):
			imgc[x, y] = np.fabs(np.sin(x/Q) + np.sin(y/Q))
	return imgc

# Funcao 3 para preenchimento da imagem cena
def f3(imgc, C, Q):
	for x in range(C):
		for y in range(C):
			imgc[x, y] = np.fabs((x/Q) - np.sqrt(y/Q))
	return imgc

# Funcao 4 para preenchimento da imagem cena
def f4(imgc, C):
	for x in range(C):
		for y in range(C):
			imgc[x, y] = random.random()
	return imgc

# Funcao 5 para preenchimento da imagem cena
def f5(imgc, C):
	# Numero total de passos
	max_moves = int((C*C)/2)

	# Passo 1
	x = y = 0
	imgc[x, y] = 1;

	for i in range(max_moves):
		# Passo na direcao x
		dx = random.randint(-1, 1)
		x = (x + dx) % C
		imgc[x, y] = 1

		# Passo na direcao y
		dy = random.randint(-1, 1)
		y = (y + dy) % C
		imgc[x, y] = 1

	return imgc

# Funcao para realizar a normalizacao
def f_norm(img):
	img_max = np.max(img)
	img_min = np.min(img)
	img_norm = (img - img_min) / (img_max - img_min)
	return img_norm

# Funcao para realizar o calculo do maximo local
def f_max_local(imgc, x, y, d):
	return imgc[(x * d):(x * d) + d, (y * d):(y * d) + d].max()

# Funcao para realizada da quantizacao e amostragem
def amost_quant(g, imgc, C, N, B):
	# Quantidades de pixel na vertical e horizontal a serem considerados
	d = int(C / N)

	# Amostragem
	for x in range(N):
		for y in range(N):
			g[x, y] = f_max_local(imgc, x, y, d)

	# Quantizacao
	g_norm = f_norm(g)
	g_norm = (g_norm * 255).astype(np.uint8)

	g = g_norm >> (8 - B)	

	return g

# Funcao para o calculo do erro
def RMSE(g, imgr, N):
	sum = 0
	for i in range(N):
		for j in range(N):
			sum += (int(g[i, j]) - int(imgr[i, j])) ** 2
	return np.sqrt(sum)

def main():
	# Nome do arquivo com a imagem referencia
	filename = str(input()).rstrip()
	# Tamanho lateral da imagem da cena
	C = int(input())
	# Tipo de funcao a ser utilizada
	f = int(input())
	# Parametro para as funcoes 2 e 3
	Q = int(input())
	# Tamanho lateral da imagem digital
	N = int(input())
	# Numero de bits por pixel na etapa de quantizacao
	B = int(input())
	# Semente para a funcao random
	S = int(input())
	random.seed(S)

	# Imagens a serem utilizadas
	imgc = np.zeros((C,C))
	imgd = np.zeros((N,N))
	imgr = np.load(filename)

	# Chamada para a funcao escolhida
	if f == 1:
		imgc = f1(imgc, C)
	elif f == 2:
		imgc = f2(imgc, C, Q)
	elif f == 3:
		imgc = f3(imgc, C, Q)
	elif f == 4:
		imgc = f4(imgc, C)
	elif f == 5:
		imgc = f5(imgc, C)
	else:
		return 1 

	# Apos a imagem gerada por uma das funcoes acima, realizar a normalizacao
	imgc_norm = f_norm(imgc)
	imgc_norm = (imgc_norm * 65535).astype(np.uint16)

	# Amostragem e quantizacao
	g = amost_quant(imgd, imgc_norm, C, N, B)

	# Comparacao
	erro = int(RMSE(g, imgr, N))
	print(erro)


if __name__ == "__main__":
	main()