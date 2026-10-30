import cv2
import numpy as np

capture = cv2.VideoCapture(0)

while True:
    _, frame = capture.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Conversão de cor RGB pra HSV

    limite_inferior = np.array([0, 70, 50]) #Cria um range de cores inferior e superior p/ amarelo
    limite_superior = np.array([10, 255, 255])

    mascara = cv2.inRange(frame_hsv, limite_inferior, limite_superior) #Aplica uma mascara para ser aplicada no frame_hsv, de forma que se essa cor for encontrada ela ficará branca e o resto preto.
    contornos,_ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Aqui ele encontra os contornos para encontrar exatamente o ponto específico (x,y)

    for contorno in contornos:
       if cv2.contourArea(contorno) > 500:   #Evita o ruído, ele só identifica a área grandes.
            x, y, w, h = cv2.boundingRect(contorno) #Pega o contorno e armazena em 4 números - x e y posição e w e h altura e largura.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Mostra um retangulo no frame ao redor do objeto do frame para que seja possível encontrar a posição específica.
            cv2.putText(frame, f"X: {x} Y: {y}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    key = cv2.waitKey(1)
    if key == 27:
        break

    cv2.imshow('Imagem', frame) #Aqui ele abre o frame normal no formato rgb
    cv2.imshow('Mask', mascara) #Aqui ele abre o frame da mask... branco se o objeto for identificado.

capture.release()
cv2.destroyAllWindows()