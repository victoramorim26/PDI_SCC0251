#	Author: Victor de A. Amorim		NUSP: 9277642
#	SCC0251 | 2018_1Sem
#	Trabalho 2 - Realce e Superresolução

import numpy as np
import imageio
import sys

# Funcao para o calculo do histograma de cada imagem: L1, L2, L3, L4
def hist(L1, L2, L3, L4):
	# Dimensao obtida a partir da L1, ja que as 4 imagens tem a mesma dimensao
	N, M = L1.shape
	
	# Criação do histrograma para cada imagem
	h1 = np.zeros(256).astype(int)
	h2 = np.zeros(256).astype(int)
	h3 = np.zeros(256).astype(int)
	h4 = np.zeros(256).astype(int)
	
	# Adiciona a ocorrencia diretamento no array do histograma correspondente
	for i in range(N):
		for j in range(M):
			h1[L1[i, j]] += 1
			h2[L2[i, j]] += 1
			h3[L3[i, j]] += 1
			h4[L4[i, j]] += 1

	return h1, h2, h3, h4

# Funcao para a equalizacao dos histogramas separados
def hist_eq(L1, h_L1, L2, h_L2, L3, h_L3, L4, h_L4):
	# Dimensao obtida a partir da L1, ja que as 4 imagens tem a mesma dimensao
	N, M = L1.shape
	
	# Criação das imagens de saida
	L1_out = np.zeros((N, M)).astype(int)
	L2_out = np.zeros((N, M)).astype(int)
	L3_out = np.zeros((N, M)).astype(int)
	L4_out = np.zeros((N, M)).astype(int)

	# Criação do histrograma acumulado para cada imagem
	hc_L1 = np.zeros(256).astype(int)
	hc_L2 = np.zeros(256).astype(int)
	hc_L3 = np.zeros(256).astype(int)
	hc_L4 = np.zeros(256).astype(int)

	# Calculo dos histogramas aculumados
	hc_L1[0] = h_L1[0]
	hc_L2[0] = h_L2[0]
	hc_L3[0] = h_L3[0]
	hc_L4[0] = h_L4[0]
	for i in range(1, 256):
		hc_L1[i] = h_L1[i] + hc_L1[i - 1]
		hc_L2[i] = h_L2[i] + hc_L2[i - 1]
		hc_L3[i] = h_L3[i] + hc_L3[i - 1]
		hc_L4[i] = h_L4[i] + hc_L4[i - 1]

	# Calculo para a equalizacao das imagens
	for r in range(256):
		s1 = (255 / float(N * M)) * hc_L1[r]
		s2 = (255 / float(N * M)) * hc_L2[r]
		s3 = (255 / float(N * M)) * hc_L3[r]
		s4 = (255 / float(N * M)) * hc_L4[r]
		# Encontra na imagem a ocorrencia dos valores 'r' e altera para o 's' correspondente
		L1_out[np.where(L1 == r)] = s1
		L2_out[np.where(L2 == r)] = s2
		L3_out[np.where(L3 == r)] = s3
		L4_out[np.where(L4 == r)] = s4

	return L1_out, L2_out, L3_out, L4_out

# Funcao para o calculo do histograma de todas as imagens: L1, L2, L3, L4
def hist_all(L1, L2, L3, L4):
	# Dimensao obtida a partir da L1, ja que as 4 imagens tem a mesma dimensao
	N, M = L1.shape
	
	# Criação do histrograma
	h = np.zeros(256).astype(int)
	
	# Adiciona a ocorrencia diretamento no array do histograma
	for i in range(N):
		for j in range(M):
			h[L1[i, j]] += 1
			h[L2[i, j]] += 1
			h[L3[i, j]] += 1
			h[L4[i, j]] += 1

	# O histograma deve apresentar a media das ocorrencias nas imagens		
	h = h/4
	return h

# Funcao para a equalizacao do histograma geral
def	hist_eq_all(L1, L2, L3, L4, h):
	# Dimensao obtida a partir da L1, ja que as 4 imagens tem a mesma dimensao
	N, M = L1.shape
	
	# Criação das imagens de saida
	L1_out = np.zeros((N, M)).astype(int)
	L2_out = np.zeros((N, M)).astype(int)
	L3_out = np.zeros((N, M)).astype(int)
	L4_out = np.zeros((N, M)).astype(int)

	# Criação do histrograma acumulado
	hc = np.zeros(256).astype(int)
	
	# Calculo do histograma aculumado
	hc[0] = h[0]
	for i in range(1, 256):
		hc[i] = h[i] + hc[i - 1]

	# Calculo para a equalizacao das imagens
	for r in range(256):
		s = (255 / float(N * M)) * hc[r]
		# Encontra na imagem a ocorrencia dos valores 'r' e altera para o 's'
		L1_out[np.where(L1 == r)] = s
		L2_out[np.where(L2 == r)] = s
		L3_out[np.where(L3 == r)] = s
		L4_out[np.where(L4 == r)] = s

	return L1_out, L2_out, L3_out, L4_out

# Funcao para realizar o Ajuste Gamma
def gamma(L1, L2, L3, L4, g):
	# Dimensao obtida a partir da L1, ja que as 4 imagens tem a mesma dimensao
	N, M = L1.shape

	# Criação das imagens de saida
	L1_out = np.zeros((N, M)).astype(int)
	L2_out = np.zeros((N, M)).astype(int)
	L3_out = np.zeros((N, M)).astype(int)
	L4_out = np.zeros((N, M)).astype(int)

	# Ajusta cada posicao da imagem
	for i in range(N):
		for j in range(M):
			L1_out[i, j] = np.floor((255 * ((L1[i, j]/255.0) ** (1/g))))
			L2_out[i, j] = np.floor((255 * ((L2[i, j]/255.0) ** (1/g))))
			L3_out[i, j] = np.floor((255 * ((L3[i, j]/255.0) ** (1/g))))
			L4_out[i, j] = np.floor((255 * ((L4[i, j]/255.0) ** (1/g))))

	return L1_out, L2_out, L3_out, L4_out

# Funcao para realizar a superresolucao
def superres(L1, L2, L3, L4):
	# Dimensao obtida a partir da L1, ja que as 4 imagens tem a mesma dimensao
	N, M = L1.shape

	# Imagem gerada a partir da superresolucao
	H = np.zeros((2*N, 2*M)).astype(int)

	# Percorre-se a imagem gerada em busca dos valores a serem preenchidos
	for i in range(2 * N):
		for j in range (2 * M):
			if i % 2 == 0:
				if j % 2 == 0: # L1
					H[i, j] = L1[int(i/2), int(j/2)]
				else: # L2
					H[i, j] = L2[int(i/2), int(j/2)]
			else:
				if j % 2 == 0: # L3
					H[i, j] = L3[int(i/2), int(j/2)]
				else: #L4
					H[i, j] = L4[int(i/2), int(j/2)]

	return H

# Funcao para o calculo do erro
def RMSE(H, H_out):
	# Dimensao obtida a partir da H
	N, M = H.shape

	# Calculo do erro
	count = 0
	for i in range(N):
		for j in range(M):
			count += (int(H[i, j]) - int(H_out[i, j])) ** 2
			
	return np.sqrt(count/(N*M))

def main():
	# Nomes dos arquivos base
	imglow_fn = str(input()).rstrip()
	L1_fn = imglow_fn + "1.png"
	L2_fn = imglow_fn + "2.png"
	L3_fn = imglow_fn + "3.png"
	L4_fn = imglow_fn + "4.png"
	imghigh_fn = str(input()).rstrip()
	H_fn = imghigh_fn + ".png"

	# Imagens a serem utilizadas
	L1 = imageio.imread(L1_fn)
	L2 = imageio.imread(L2_fn)
	L3 = imageio.imread(L3_fn)
	L4 = imageio.imread(L4_fn)
	H = imageio.imread(H_fn)

	# Opcao de realce
	op_realce = int(input())
	g = float(input())

	# Chamada para a funcao escolhida
	if op_realce == 0: # Não realiza realce
		L1_out = L1
		L2_out = L2
		L3_out = L3
		L4_out = L4
	elif op_realce == 1: # Histograma acumulado de cada imagem
		h_L1, h_L2, h_L3, h_L4 = hist(L1, L2, L3, L4)
		L1_out, L2_out, L3_out, L4_out = hist_eq(L1, h_L1, L2, h_L2, L3, h_L3, L4, h_L4)
	elif op_realce == 2: # Histograma acumulado de todas as imagens
		h_L = hist_all(L1, L2, L3, L4)
		L1_out, L2_out, L3_out, L4_out = hist_eq_all(L1, L2, L3, L4, h_L)
	elif op_realce == 3: # Ajuste Gamma
		L1_out, L2_out, L3_out, L4_out = gamma(L1, L2, L3, L4, g)
	else:
		print("Entrada invalida!")
		return 1

	#Superresolução
	H_out = superres(L1_out, L2_out, L3_out, L4_out)

	# Comparacao
	erro = RMSE(H, H_out)
	print(np.round(erro, 4))


if __name__ == "__main__":
	main()