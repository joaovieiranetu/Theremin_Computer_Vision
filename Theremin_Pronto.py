
import cv2
import numpy as np
import sounddevice as sd

# Nomes das notas musicais para mostrar na tela
nomes_notas = ["Do", "Re", "Mi", "Fa", "Sol", "La", "Si", "Do+"]
Notas = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25] #C,D,E,F,G,A,B,C+

# Variáveis globais para controlar o áudio
frequencia_atual = 440.0
volume_atual = 0.0

# Variáveis utilizadas para suavizar a frequência e o volume
frequencia_suave = 440.0
volume_suave = 0.0

fase = 0.0
amostragem = 44100

# Controla a velocidade da suavização da posição do objeto
# Quanto maior o valor, mais rápido o volume acompanha o movimento
suavizacao_posicao = 0.35

# Controla a velocidade da suavização do áudio
tempo_volume = 0.025
tempo_frequencia = 0.025

# Armazena a posição Y suavizada do objeto
posicao_y_suave = None


# Esta função é chamada automaticamente pelo SoundDevice.
# Ela gera as amostras da onda sonora para manter o áudio contínuo.
def audio_callback(outdata, frames, time, status):
    global fase
    global frequencia_suave
    global volume_suave

    # Calcula os valores utilizados na suavização do som
    # Esses valores controlam a velocidade com que a frequência
    # e o volume chegam aos valores desejados
    alpha_volume = 1 - np.exp(-1 / (amostragem * tempo_volume))
    alpha_frequencia = 1 - np.exp(-1 / (amostragem * tempo_frequencia))

    # Cria um array vazio para armazenar as amostras de áudio
    buffer = np.zeros(frames)

    # Gera a onda sonora amostra por amostra
    for i in range(frames):

        # Suaviza a frequência para evitar mudanças bruscas nas notas
        frequencia_suave += alpha_frequencia * (frequencia_atual - frequencia_suave)

        # Suaviza o volume para evitar mudanças bruscas na amplitude
        volume_suave += alpha_volume * (volume_atual - volume_suave)

        # Incrementa a fase da onda de acordo com a frequência
        fase += 2 * np.pi * frequencia_suave / amostragem

        # Mantém a fase entre 0 e 2*PI para evitar valores muito grandes
        if fase >= 2 * np.pi:
            fase -= 2 * np.pi

        # Gera a onda senoidal e multiplica pela amplitude atual
        buffer[i] = np.sin(fase) * volume_suave

    # Transforma o áudio mono em estéreo
    # Envia o mesmo sinal para os dois canais
    outdata[:] = np.column_stack((buffer, buffer))


# Função responsável por atualizar a frequência e o volume do som
def gerar_som(frequencia, volume=0.5):
    global frequencia_atual
    global volume_atual

    frequencia_atual = frequencia

    # Garante que o volume fique entre 0 e 1
    volume_atual = np.clip(volume, 0, 1)


# Inicializa o fluxo de áudio apenas uma vez
stream = sd.OutputStream(
    samplerate=amostragem,
    channels=2,
    callback=audio_callback,
    blocksize=256
)

stream.start()

# Inicializa a câmera
capture = cv2.VideoCapture(0)

while True:

    # Captura um frame da câmera
    ok, frame = capture.read()

    if not ok:
        break

    # Obtém a altura e a largura do frame
    altura_frame, largura_frame, _ = frame.shape

    # Inverte a imagem horizontalmente para acompanhar o movimento do objeto
    frame = cv2.flip(frame, 1)

    # Converte a imagem de BGR para HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define os limites inferior e superior para identificar a cor verde
    limite_inferior = np.array([35, 50, 50])
    limite_superior = np.array([85, 255, 255])

    # Cria uma máscara onde a cor verde fica branca e o restante preto
    mascara = cv2.inRange(frame_hsv, limite_inferior, limite_superior)

    # Cria uma matriz utilizada para remover pequenos ruídos da máscara
    kernel = np.ones((5, 5), np.uint8)

    # Remove pequenos pontos brancos que podem atrapalhar a detecção
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    # Preenche pequenas falhas dentro do objeto identificado
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    # Encontra os contornos dos objetos verdes identificados
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Variável para verificar se algum objeto foi encontrado
    encontrou_objeto = False

    # Verifica se existem contornos na imagem
    if len(contornos) > 0:

        # Seleciona o maior contorno encontrado
        # Isso evita que pequenos objetos verdes sejam utilizados
        contorno = max(contornos, key=cv2.contourArea)

        # Ignora objetos com áreas menores que 500 pixels
        if cv2.contourArea(contorno) > 500:

            # Encontra a posição e o tamanho do retângulo do objeto
            x, y, w, h = cv2.boundingRect(contorno)

            # Calcula o centro do objeto utilizando o retângulo
            cx = x + (w // 2)
            cy = y + (h // 2)

            # Normaliza a posição X do objeto entre 0 e 1
            valor_normal_x = cx / largura_frame

            # Normaliza a posição Y do objeto entre 0 e 1
            # O valor é invertido porque o eixo Y da imagem cresce para baixo
            # Assim, quanto mais alto o objeto estiver, maior será o volume
            valor_normal_y = 1.0 - (cy / altura_frame)

            # Lógica das notas musicais
            # Divide a largura do frame em 8 partes
            # Cada parte corresponde a uma nota musical
            posicao_nota = int(valor_normal_x * len(Notas))

            # Impede que o índice ultrapasse a quantidade de notas
            if posicao_nota >= len(Notas):
                posicao_nota = len(Notas) - 1

            if posicao_nota < 0:
                posicao_nota = 0

            # Seleciona a frequência e o nome da nota musical
            frequencia = Notas[posicao_nota]
            nome_da_nota = nomes_notas[posicao_nota]

            # Garante que a posição Y esteja entre 0 e 1
            valor_normal_y = np.clip(valor_normal_y, 0, 1)

            # Na primeira detecção, utiliza a posição atual do objeto
            if posicao_y_suave is None:
                posicao_y_suave = valor_normal_y

            else:
                # Suaviza a posição Y para evitar oscilações no volume
                # A posição suavizada se aproxima gradualmente da posição real
                posicao_y_suave += suavizacao_posicao * (valor_normal_y - posicao_y_suave)

            # A amplitude será definida pela posição vertical suavizada
            volume_atual_frame = posicao_y_suave

            # Atualiza a frequência e o volume do som
            gerar_som(frequencia, volume=volume_atual_frame)

            # Desenha um círculo vermelho no centro do objeto
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Desenha um retângulo verde ao redor do objeto identificado
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Mostra o nome da nota musical na tela
            cv2.putText(frame, f"Nota: {nome_da_nota}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            encontrou_objeto = True

    # Se o objeto não for encontrado, o volume diminui suavemente até zero
    if not encontrou_objeto:
        gerar_som(frequencia_atual, volume=0)

        # Reinicia a posição suavizada para a próxima detecção
        posicao_y_suave = None

    # Mostra o frame original com as informações do objeto
    cv2.imshow('Imagem', frame)

    # Mostra a máscara utilizada para identificar a cor verde
    cv2.imshow('Mask', mascara)

    # Verifica se a tecla ESC foi pressionada
    key = cv2.waitKey(1)

    if key == 27:
        break

# Libera a câmera
capture.release()

# Fecha todas as janelas do OpenCV
cv2.destroyAllWindows()

# Encerra o fluxo de áudio
stream.stop()
stream.close()
