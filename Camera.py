import cv2

webcam = cv2.VideoCapture(0) #Aqui ele faz a captura da webcam

while True:
    ok, frame = webcam.read()
    if not ok:
        break
    
    key = cv2.waitKey(2) #Esse waitkey está servindo para renderizar o vídeo (montar o loop dos frames)
    cv2.imshow('imagem', frame) #Aqui ele aprensenta a imagem quando o código é executado.

    if key == 27: #Aqui quando o usuário pressiona o esc o frame é quebrado automaticamente
        break


webcam.release()
cv2.destroyAllWindows()
