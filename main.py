from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from tensorflow.keras.preprocessing.image import ImageDataGenerator #
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from tensorflow.keras.preprocessing import image

train_dir = 'train/'
test_dir = 'test/'

def learn():
    train_datagen = ImageDataGenerator(rescale=1. / 255)  # Нормализация изображений
    test_datagen = ImageDataGenerator(rescale=1. / 255)  # Нормализация изображений

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='sparse')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='sparse')

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='linear')  # Используем линейную активацию для регрессии
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    history = model.fit(
        train_generator,
        steps_per_epoch=100,  # Количество шагов за эпоху
        epochs=10,  # Количество эпох
        validation_data=test_generator,
        validation_steps=50)  # Количество шагов валидации

    test_loss, test_acc = model.evaluate(test_generator, steps=50)
    print('test mae:', test_acc)

    return model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # Например, 16 мегабайтов
CORS(app)

model = learn()

@app.route('/predict', methods=['POST'])
def predict():
    # Проверяем, существует ли файл в отправленной форме
    if 'file' not in request.files:
        return jsonify({'error': 'Нет файла'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    if file:
        filepath = 'newimg/' + file.filename
        file.save(filepath)
    else:
        return jsonify({'error': 'Ошибка при загрузке файла'}), 500

    test_image_path = 'newimg/' + file.filename
    test_image = image.load_img(test_image_path, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)  # Преобразование к форме, подходящей для модели

    predict_temp = model.predict(test_image)

    if predict_temp:
        return jsonify({'message': f'Температура "{predict_temp}" условных единиц'}), 200

    print('Predicted temperature:', predict_temp)

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)