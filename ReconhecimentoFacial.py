import cv2
from deepface import DeepFace

# Carregar o classificador Haar Cascade para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a captura de vídeo (webcam)
cap = cv2.VideoCapture(0)

while True:
    # Captura frame a frame
    ret, frame = cap.read()

    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Para cada rosto detectado
    for (x, y, w, h) in faces:
        # Desenhar um retângulo ao redor do rosto detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recortar a região do rosto detectado
        face_roi = frame[y:y + h, x:x + w]

        # Usar DeepFace para identificar a emoção facial
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        print(result)
        # Obter a emoção mais provável
        dominant_emotion = result[0]['dominant_emotion']

        # Colocar o nome da emoção na tela
        cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Exibir o vídeo com as deteções e emoções
    cv2.imshow('Facial Expression Recognition', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a captura e fechar as janelas
cap.release()
cv2.destroyAllWindows()
