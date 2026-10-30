import cv2
import numpy as np
import pygame

# Configuração inicial do áudio
pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()

Notas = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25] #C,D,E,F,G,A,B,C+

def gerar_som(frequencia, volume=0.5): #Função de gerar som baseada no pygame
    amostragem = 44100
    duracao = 0.1
    t = np.linspace(0, duracao, int(amostragem * duracao), False)
    
    onda_mono = np.sin(2 * np.pi * frequencia * t) * volume * 32767
    onda_int = onda_mono.astype(np.int16)
    onda_stereo = np.stack((onda_int, onda_int), axis=-1)
    
    return pygame.sndarray.make_sound(onda_stereo)

capture = cv2.VideoCapture(0)

# Variável para controlar o som atual e evitar sobreposição
meu_som = None

while True:
    ok, frame = capture.read()
    if not ok:
        break
    
    altura_frame, largura_frame, _ = frame.shape
    frame = cv2.flip(frame, 1)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Conversão de cor RGB pra HSV

    limite_inferior = np.array([35, 50, 50]) #Cria um range de cores inferior e superior p/ amarelo
    limite_superior = np.array([85, 255, 255])

    mascara = cv2.inRange(frame_hsv, limite_inferior, limite_superior) #Aplica uma mascara para ser aplicada no frame_hsv, de forma que se essa cor for encontrada ela ficará branca e o resto preto.
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Aqui ele encontra os contornos para encontrar exatamente o ponto específico (x,y)

    for contorno in contornos:
        if cv2.contourArea(contorno) > 500:   #Evita o ruído, ele só identifica a área grandes.
            x, y, w, h = cv2.boundingRect(contorno) #Pega o contorno e armazena em 4 números - x e y posição e w e h altura e largura.

            cx = x + (w//2) #Encontrando o meio real do frame - inicio x + a largura dividida por 2
            cy = y + (h//2)

            valor_normal_x = cx / largura_frame 
            valor_normal_y = 1.0 - (cy / altura_frame)

            # Lógica das Notas Musicais
            posicao_nota = int((valor_normal_x) * len(Notas))
            if posicao_nota >= len(Notas): posicao_nota = len(Notas) - 1
            if posicao_nota < 0: posicao_nota = 0
            
            frequencia = Notas[posicao_nota]
            volume_atual = np.clip(valor_normal_y, 0, 1) # Garante que o volume fique entre 0 e 1
            
            # Gerenciamento do som para evitar ruído excessivo
            if meu_som is not None:
                meu_som.stop()

            meu_som = gerar_som(frequencia, volume = volume_atual) #crio um objeto de som, que vai chamar a função gerar som por meio da frequencia.
            meu_som.play()

            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Mostra um retangulo no frame ao redor do objeto do frame para que seja possível encontrar a posição específica.
            cv2.putText(frame, f"X: {x} Y: {y}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Imagem', frame) #Aqui ele abre o frame normal no formato rgb
    cv2.imshow('Mask', mascara) #Aqui ele abre o frame da mask... branco se o objeto for identificado.

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
pygame.quit()