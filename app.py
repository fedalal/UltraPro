

import streamlit as st
from PIL import Image 
from segment import process

st.title('Alexander Feduleev detect object')

image_file = st.file_uploader('Load an image', type=['png', 'jpg'])  # Добавление загрузчика файлов

if not image_file is None:                                           # Выполнение блока, если загружено изображение
    col1, col2 = st.beta_columns(2)                                  # Создание 2 колонок
    image = Image.open(image_file)                                   # Открытие изображения
    results = process(image_file)                                    # Обработка изображения с помощью функции, реализованной в другом файле
    col1.text('Source image')
    col1.image(results[1])                                           # Вывод в первой колонке уменьшенного исходного изображения
    col2.text(results[2])
    col2.image(results[0])                                           # Вывод маски второй колонке
    
