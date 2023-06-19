import tensorflow as tf
import io
import numpy as np

from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler

model = tf.keras.models.load_model('models/modelo_cnn_cifar100_c10.h5')

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        print("Peticion recibida")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        binary_image = post_data.split(b'\r\n\r\n')[1]

        # Decodificar imagen a PIL
        pil_image = Image.open(io.BytesIO(binary_image))
        
        # Transformar a 32x32
        resized_image = pil_image.resize((32, 32))

        # Convertir imagen PIL a numpy array
        np_image = np.array(resized_image)

        # Normalizar
        np_image = np_image / 255.0

        # Generar prediccion
        y_pred = model.predict(np_image.reshape(1, 32, 32, 3))
        clases = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle'
        ]

        max = 0
        for i in range(100):
            if y_pred[0][i] > max:
                max = y_pred[0][i]
                index = i

        print('Prediccion: ', clases[index])

        # Generar respuesta a la petición HTTP
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(bytes(clases[index], 'utf-8'))


print("Iniciando el servidor...")
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
server.serve_forever()
