import cv2
import numpy as np
import sounddevice as sd

# Variáveis globais do áudio
frequencia_atual = 440.0
volume_atual = 0.5

# Variáveis suavizadas
frequencia_suave = 440.0
volume_suave = 0.5

fase = 0.0
amostragem = 44100

#Esta função é chamada automaticamente pela placa de som milhares de vezes por segundo.
#Preenche o buffer de saída com os dados da onda senoidal de forma contínua ao invés de cortar e começar outra.

def audio_callback(outdata, frames, time, status):  #time e status são argumentos da função que mesmos que não use, devem estar no parametro.
    global fase
    global frequencia_suave
    global volume_suave

    # Cria um array vazio para armazenar as amostras deste bloco (frame)
    buffer = np.zeros(frames)

    # Suavização da frequência e volume: evita "pulos" bruscos no som.
    # Em vez de mudar instantaneamente, os valores deslizam 3% em direção ao alvo a cada bloco.
    frequencia_suave += 0.03 * (frequencia_atual - frequencia_suave)
    volume_suave += 0.03 * (volume_atual - volume_suave)

    # Gera a onda amostra por amostra para garantir que a fase seja contínua
    for i in range(frames):
        # Incrementa a fase baseado na frequência suavizada
        fase += 2 * np.pi * frequencia_suave / amostragem

        # Mantém a fase dentro do ciclo de 0 a 2*PI para evitar overflow numérico
        if fase > 2 * np.pi:
            fase -= 2 * np.pi

        # Calcula o valor da onda senoidal e aplica o volume
        buffer[i] = np.sin(fase) * volume_suave

    # Transforma o buffer mono em stereo (colunas esquerda e direita iguais)
    outdata[:] = np.column_stack((buffer, buffer))


def gerar_som(frequencia, volume=0.5): #Função de gerar som baseada no sounddevice
    global frequencia_atual
    global volume_atual

    frequencia_atual = frequencia
    volume_atual = volume


# Inicializa o stream apenas uma vez
stream = sd.OutputStream(
    samplerate=amostragem,
    channels=2,
    callback=audio_callback,
    blocksize=256
)

stream.start()

capture = cv2.VideoCapture(0)

# Variável para controlar o som atual
meu_som = None

while True:
    ok, frame = capture.read()
    if not ok:
        break

    altura_frame, largura_frame, _ = frame.shape
    frame = cv2.flip(frame, 1)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Conversão de cor RGB pra HSV

    limite_inferior = np.array([100, 150, 0]) #Cria um range de cores inferior e superior p/ verde
    limite_superior = np.array([140, 255, 255])

    mascara = cv2.inRange(frame_hsv, limite_inferior, limite_superior) #Aplica uma mascara para ser aplicada no frame_hsv, de forma que se essa cor for encontrada ela ficará branca e o resto preto.

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Aqui ele encontra os contornos para encontrar exatamente o ponto específico (x,y)

    encontrou_objeto = False

    for contorno in contornos:
        if cv2.contourArea(contorno) > 500:   #Evita o ruído, ele só identifica a área grandes.
            x, y, w, h = cv2.boundingRect(contorno) #Pega o contorno e armazena em 4 números - x e y posição e w e h altura e largura.

            cx = x + (w//2) #Encontrando o meio real do frame - inicio x + a largura dividida por 2
            cy = y + (h//2)

            valor_normal_x = cx / largura_frame

            # Frequência contínua baseada na posição do objeto no eixo X
            frequencia = 220 + (valor_normal_x * 660)

            # Volume fixo apenas para manter o som ativo
            volume_atual_frame = 0.5

            # Gerenciamento do som usando sounddevice
            gerar_som(frequencia, volume=volume_atual_frame)

            cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Mostra um retangulo no frame ao redor do objeto do frame para que seja possível encontrar a posição específica.
            cv2.putText(frame, f"X: {x} Y: {y}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            encontrou_objeto = True
            break

    # Se perder o objeto, reduz o volume suavemente
    if not encontrou_objeto:
        gerar_som(frequencia_atual, volume=0)

    cv2.imshow('Imagem', frame) #Aqui ele abre o frame normal no formato rgb
    cv2.imshow('Mask', mascara) #Aqui ele abre o frame da mask... branco se o objeto for identificado.

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()

stream.stop()
stream.close()