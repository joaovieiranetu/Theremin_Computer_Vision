import cv2
import numpy as np

capture = cv2.VideoCapture(0)

while True:
    _, frame = capture.read()
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Conversão de cor RGB pra HSV

    limite_inferior = np.array([35, 40, 40]) #Cria um range de cores inferior e superior p/ amarelo
    limite_superior = np.array([85, 255, 255])

    mascara = cv2.inRange(frame_hsv, limite_inferior, limite_superior) #Aplica uma mascara para ser aplicada no frame_hsv, de forma que se essa cor for encontrada ela ficará branca e o resto preto.


    key = cv2.waitKey(1)
    if key == 27:
        break

    cv2.imshow('Imagem', frame) #Aqui ele abre o frame normal no formato rgb
    cv2.imshow('Mask', mascara) #Aqui ele abre o frame da mask... branco se o objeto for identificado.

capture.release()
cv2.destroyAllWindows()