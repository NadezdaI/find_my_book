import streamlit as st
from PIL import Image
from pathlib import Path
import base64
import io
import os
import random
import pandas as pd
from functions import display_book
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEndpointEmbeddings
#from pathlib import Path

st.cache_data.clear()
st.cache_resource.clear()

st.set_page_config(layout="wide")
 
# ------ Инициализация состояния -----
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "n_books" not in st.session_state:
    st.session_state.n_books = 10

# ------------- Заголовок ---------
BASE_DIR = Path(__file__).parent

img_path = BASE_DIR / "black_white.png"
img = Image.open(img_path).convert("RGBA")

buffered = io.BytesIO()
img.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

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
Сервис предлагает интеллектуальный поиск книг по смыслу аннотаций. Мы собрали коллекцию из более чем 28 000 книг и используем методы обработки естественного языка (NLP), чтобы находить произведения, близкие по содержанию к вашему запросу.

"""
)
st.markdown("---")

# -------- Поле запроса -----------
query = st.text_input("", placeholder="Опишите, какую книгу вы ищете (например, «о философии науки и её влиянии на политику»)")

st.markdown("""
    <style>
    .stButton>button {
        background-color: #A42921;
        color: white;
        border-radius: 8px;
        height: 40px;
        width: 200px;
        font-size:16px;
    }
    .stButton>button:hover {
        background-color: #d4352c;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------- Кнопка Отправить запрос -------------
trigger = st.button("Отправить запрос")

st.markdown("""
<style>
details[open] summary { 
    display: none; 
}
</style>
""", unsafe_allow_html=True)

# ----------- Подключение к облаку Qadrant -------------

def get_secret(key: str) -> str:
    return st.secrets.get(key, os.getenv(key))

QDRANT_URL_N = get_secret("QDRANT_URL_N")
QDRANT_API_N = get_secret("QDRANT_API_N")

client = QdrantClient(url=QDRANT_URL_N, api_key=QDRANT_API_N)

embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}  
    )

# embeddings_model_API = HuggingFaceEndpointEmbeddings(
#         repo_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
#         huggingfacehub_api_token=st.secrets["HF_API_KEY"]
#     )

vector_store = QdrantVectorStore(
    client=client,
    collection_name="books_collection",
    embedding=embeddings_model
)

# ---------- Обработка запроса -------------
if trigger and query:
    st.session_state.show_results = True
    st.session_state.search_query = query
    st.session_state.search_results = vector_store.similarity_search_with_score(query, k=50)

# ---------- Отображение результатов или стартовой страницы -------------
if st.session_state.show_results and hasattr(st.session_state, 'search_results'):
    # ---------- РЕЖИМ РЕКОМЕНДАЦИЙ -----------
    st.markdown(f"### Топ 10 рекомендаций по запросу ")  # \"{st.session_state.search_query}\"
    results_to_show = st.session_state.search_results[:st.session_state.n_books]
    
    for i, (doc, score) in enumerate(results_to_show):
        title = doc.metadata.get('title', 'Не указано')
        author = doc.metadata.get('author', 'Не указано')
        link = doc.metadata.get('page_url', 'Не указано')
        img = doc.metadata.get('image_url', '')
        annotation = doc.metadata.get('annotation', 'Нет описания')
        score_formatted = f'{score:.4f}'
        
        display_book(
            title=title,
            author=author,
            link=link,
            img=img,
            annotation=annotation,
            score=score_formatted
        )

else:
    #st.markdown("""СТАРТОВАЯ СТРАНИЦА""")
    # ---------- СТАРТОВАЯ СТРАНИЦА -----------   
    df = pd.read_csv('../data/books.csv')

    def show_books(n_books):
        st.markdown(f"### Актуальные книги сегодня ")
        sampled_idx = random.sample(range(len(df)), k=n_books)

        for idx in sampled_idx:
            row = df.loc[idx]
            
            # Используем общую функцию
            display_book(
                title=row['title'],
                author=row['author'],
                link=row['link'],
                img=row['image'],
                annotation=row['annotation']
            )

    show_books(int(st.session_state.n_books))

# ---------- ОБЩИЙ БЛОК "ПОКАЗЫВАТЬ НА СТРАНИЦЕ" (отображается всегда) -----------
st.markdown("---")
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