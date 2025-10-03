import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from PIL import Image
import base64
import io
import random
import pandas as pd
import html

st.set_page_config(layout="wide")
img = Image.open("black_white.png").convert("RGBA")

# Конвертируем в Base64
buffered = io.BytesIO()
img.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

#st.title("Умный поиск книг")

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{img_str}" width="80" style="margin-right: 15px;">
        <h1 style="color:black; font-family:Verdana, Geneva, sans-serif; font-size:40px; margin:0;">
            Умный поиск книг
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
Этот сервис помогает находить книги не только по названию или автору, но и по содержанию аннотаций.
Мы собрали большую коллекцию аннотаций (28000+ книг) и используем методы обработки естественного языка (NLP), чтобы анализировать текст запроса и тексты книг.

"""
)
st.markdown("---")

query = st.text_input("", placeholder="Введите запрос (например, «Хочу книгу о философии науки и её влиянии на политику»)")
# синий #2c77d4  красный #A42921
st.markdown("""
    <style>
    .stButton>button {
        background-color: #A42921; /* зелёный */
        color: white;
        border-radius: 8px;
        height: 40px;
        width: 200px;
        font-size:16px;
    }
    .stButton>button:hover {
        background-color: #d4352c; /* тёмно-зелёный при наведении */
    }
    </style>
    """, unsafe_allow_html=True)


trigger = st.button("Отправить запрос")

if trigger:
    # здесь обработка AI
    st.write("AI ответ:", query[::-1])  # пример: переворачиваем текст

st.markdown("""
<style>
details[open] summary { 
    display: none; 
}
</style>
""", unsafe_allow_html=True)


df = pd.read_csv("../data/Nadia_books.csv")

def show_books(n_books):
    # случайная выборка индексов
    sampled_idx = random.sample(range(len(df)), k=n_books)

    for idx in sampled_idx:
        row = df.loc[idx]

        desc = str(row["description"]) if pd.notna(row["description"]) else "Описание отсутствует"
        desc = desc.encode('utf-8', 'ignore').decode('utf-8')
        desc = html.escape(desc)
        
        # Разбиваем описание на слова
        words = desc.split()
        
        if len(words) > 50:
            desc_preview = " ".join(words[:50])
            desc_rest = " ".join(words[50:])
            show_details = f'<details><summary>Показать больше</summary>{desc_rest}</details>'
        else:
            desc_preview = desc
            show_details = ""

        st.markdown(f"""
        <div style="
            border: 1px solid #ddd; 
            border-radius: 10px; 
            padding: 10px; 
            margin-bottom: 15px; 
            background-color: #f9f9f9;
        ">
            <table style="border-collapse: collapse; width: 100%; border: none;">
                <tr>
                    <td style="padding: 5px; vertical-align: top; width: 120px; border: none;">
                        <a href="{row['link']}" target="_blank">
                            <img src="{row['image']}" style="width:100px; border-radius: 6px;"/>
                        </a>
                    </td>
                    <td style="padding: 5px; vertical-align: top; border: none;">
                        <b><a href="{row['link']}" target="_blank">{row['title']}</a></b><br>
                        <i>{row['author']}</i><br>
                        {desc_preview}
                        {show_details}
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# Инициализация состояния
if "n_books" not in st.session_state:
    st.session_state.n_books = 5

# Обработка нажатий кнопок внизу
if "books_updated" not in st.session_state:
    st.session_state.books_updated = False

# Сначала показываем книги
show_books(int(st.session_state.n_books))

# Разделитель
st.markdown("---")

# Нижний блок — поле ввода для количества книг
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col3:
    st.markdown(
        '<div style="text-align: right; display: flex; align-items: center; height: 100%; justify-content: flex-end; margin-top: 5px;">Показывать на странице:</div>', 
        unsafe_allow_html=True
    )
with col4:
    n_books_input = st.number_input(
        "",
        min_value=1,
        max_value=50,
        value=st.session_state.n_books,
        step=1,
        label_visibility="collapsed"
    )
    
    if n_books_input != st.session_state.n_books:
        st.session_state.n_books = n_books_input
        st.rerun()
        
        
