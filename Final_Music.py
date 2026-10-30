import cv2
import numpy as np
import pygame

pygame.mixer.pre_init(44100, -16, 1, 512)
pygame.init()

def gerar_onda_senoidal(frequencia, duracao=0.03, volume=0.3):
    amostragem = 44100
    pontos = np.linspace(0, duracao, int(amostragem * duracao), False)
    onda = np.sin(2 * np.pi * frequencia * pontos)
    onda_inteira = (onda * volume * 32767).astype(np.int16)
    
    if pygame.mixer.get_init()[2] == 2:
        # Cria duas colunas (uma para cada ouvido) com o mesmo som
        onda_inteira = np.repeat(onda_inteira[:, np.newaxis], 2, axis=1)
        return pygame.sndarray.make_sound(onda_inteira)

capture = cv2.VideoCapture(0)

while True:
    _, frame = capture.read()
    frame = cv2.flip(frame, 1)  #A função flip vai inverter a direção da câmera (ao ir para a direita o objeto seguirá)
    h, w, _ = frame.shape
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Conversão de cor RGB pra HSV

    limite_inferior = np.array([160, 50, 80]) #Cria um range de cores inferior e superior p/ rosa
    limite_superior = np.array([180, 255, 255])

    mascara = cv2.inRange(frame_hsv, limite_inferior, limite_superior) #Aplica uma mascara para ser aplicada no frame_hsv, de forma que se essa cor for encontrada ela ficará branca e o resto preto.
    contornos,_ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Aqui ele encontra os contornos para encontrar exatamente o ponto específico (x,y)

    for contorno in contornos:
        if cv2.contourArea(contorno) > 1000:   #Evita o ruído, ele só identifica a área grandes.
            x, y, w, h = cv2.boundingRect(contorno) #Pega o contorno e armazena em 4 números - x e y posição e w e h altura e largura.
            cx = x + (w // 2)
            valor_x = cx / w  

            frequencia = 110 + (valor_x * 110) # Converte para uma frequência audível (ex: 220Hz a 880Hz)   
            som = gerar_onda_senoidal(frequencia)
            som.play()
            
            valor_normalizado = cx / w #Divide a posição atual pela largura total; p/ encontrar o meio - 0.5
            cv2.circle(frame, (cx, y), 10, (0, 100, 50), -1) #Cria um círculo no ponto específico cx e cy do frame e não na mask
            cv2.putText(frame, f"Nota: {valor_normalizado:.2f}", (cx, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
   
    key = cv2.waitKey(1)
    if key == 27:
        break

    cv2.imshow('Imagem', frame) #Aqui ele abre o frame normal no formato rgb
    cv2.imshow('Mask', mascara) #Aqui ele abre o frame da mask... branco se o objeto for identificado.

capture.release()
cv2.destroyAllWindows()