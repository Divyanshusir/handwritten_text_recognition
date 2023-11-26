import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, request, jsonify
import cv2
import base64
from tensorflow.keras.layers import StringLookup
import string
from flask_cors import CORS 
import tempfile
import os



app = Flask(__name__)
CORS(app)

def create_temp_image(data_url):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image_data = data_url.split(",")[1]
        binary = base64.b64decode(image_data)
        temp_file.write(binary)
        return temp_file.name
    

class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

custom_objects = {"CTCLayer": CTCLayer}

with keras.saving.custom_object_scope(custom_objects):
    model = keras.models.load_model("C:\\Users\\nikhi\\Desktop\\college course\\semester 5\\Ai\\handwriting reconginsation\\HWR.h5")


prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)


@app.route('/recognize_text', methods=['POST'])
def recognize_text():
    try:
        data_url = request.get_json().get('image', '')
        temp_image_path = create_temp_image(data_url)
        custom_image = preprocess_image(temp_image_path)
        custom_image = custom_image.numpy().reshape((image_width, image_height, 1))
        prediction = prediction_model.predict(np.expand_dims(custom_image, axis=0))
        decoded_text = decode_batch_predictions(prediction)
        
        print(decoded_text[0])

        return jsonify({'recognizedText': decoded_text[0]})

    except Exception as e:
        return jsonify({'error': str(e)})




def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

image_width = 128
image_height = 32
def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


train_labels_cleaned = []
characters = set()
max_len = 0

characters = ['!', '"', '#', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
max_len = len(characters)
characters = sorted(list(characters))

char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text




def base64_to_image(data_url):
    try:
        image_b64 = data_url.split(",")[1]
        binary = base64.b64decode(image_b64)
        temp = np.asarray(bytearray(binary), dtype="uint8")
        image = cv2.imdecode(temp, cv2.IMREAD_COLOR)
        print(image)
        return image
    except Exception as e:
        print("Error in base64_to_image:", str(e))
        return None



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
