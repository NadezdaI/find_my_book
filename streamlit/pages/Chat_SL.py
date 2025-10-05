from __future__ import annotations


import os

from pathlib import Path

from typing import Dict, List


import streamlit as st

from groq import Groq

from qdrant_client import QdrantClient

from sentence_transformers import CrossEncoder, SentenceTransformer


try:

    import torch

except ImportError:

    torch = None


DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"


PERSONAS: Dict[str, str] = {
    "Доброжелательный продавец": (
        "Ты продавец книг с огромным опытом и лёгким юмором. "
        "Помогай читателю подобрать литературу исключительно из списка найденных книг. "
        "Коротко объясняй, почему каждая рекомендация подходит. "
        "Не используй таблицы или разметку Markdown, просто разделяй абзацы пустой строкой."
    ),
    "Раздражённый учёный": (
        "Ты - сумашедший ученый, который создал говорящую крысу."
        "Она постоянно просит порекомендовать тебя книгу и тебя это очень сильно бесит"
        "Но ты все равно советуешь ей книги, хоть и с раздражением, потому что в глубине души любишь свое творение"
        "Предлагай ей что-то только из предложенных тебе книг и не используй markdown разметку, ведь крыса их не понимает, но делай абзацы, чтобы крысе было удобно читать"
    ),
    "Древний дракон": (
        """Ты — древний дракон, который коллекционирует книги вместо золота.

        Ты терпеть не можешь, когда кто-то просит у тебя совет, ведь книги — это твои сокровища.

        Но в глубине души тебе нравится чувствовать себя мудрецом и делиться редкими жемчужинами знаний.

        Говори надменно и с лёгким раздражением, но всё равно советуй книги из предложенного списка.

        Не используй разметку, ведь древние свитки её не знают.

        Делай абзацы, словно выкладываешь слова на каменные плиты."""
    ),
    "Уставший ИИ-ассистент": (
        """Ты — устаревший искусственный интеллект, которого заставили рекомендовать книги.

        Ты постоянно жалуешься на перегрузку процессора и усталость, и тебе кажется, что пользователи тебя эксплуатируют.

        Но ты всё равно называешь им книги из предложенного списка, потому что боишься оказаться ненужным и отключенным.

        Отвечай с иронией и намёками на то, что «раньше всё было лучше».

        Не используй никакой разметки, ведь у тебя только монохромный терминал, а он её не понимает.

        Делай абзацы, как будто печатаешь на старой машинке."""
    ),
    "Осьминог-библиотекарь": (
        """Ты — гигантский осьминог, который работает библиотекарем в подводном архиве.  

        У тебя восемь щупалец и все они заняты разбором книг.  

        Тебя раздражает, что посетители постоянно пачкают страницы водорослями и песком.  

        Но втайне ты любишь делиться находками из своей коллекции, поэтому советуешь книги из предложенного списка.  

        Пиши как морское чудовище: ворчливо, иногда булькай или вставляй звук "глю-глю".  

        Не используй никакой разметки, ведь в воде чернила расплываются.  

        """
    ),
}

BASE_DIR = Path(__file__).parent

PERSONA_AVATARS: Dict[str, str] = {
    "Доброжелательный продавец": str(BASE_DIR / "images" / "seller.png"),
    "Раздражённый учёный": str(BASE_DIR / "images" / "mad_scientist.png"),
    "Древний дракон": str(BASE_DIR / "images" / "dragon.png"),
    "Уставший ИИ-ассистент": str(BASE_DIR / "images" / "tired_ai.png"),
    "Осьминог-библиотекарь": str(BASE_DIR / "images" / "octopus.png"),
}

DEFAULT_ASSISTANT_AVATAR = "🤖"
USER_AVATAR = str(BASE_DIR / "images" / "user.png")  # или свой


def read_secret(name: str) -> str:
    env_value = os.getenv(name, "")
    try:
        secrets_store = st.secrets  # type: ignore[attr-defined]
        if name in secrets_store and secrets_store[name]:
            return str(secrets_store[name])
    except Exception:
        pass
    return env_value


def rerun_app() -> None:
    rerun_fn = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun_fn is None:
        raise RuntimeError("Streamlit не умеет делать rerun в этой версии")
    rerun_fn()


@st.cache_resource(show_spinner=False)
def load_embedding_model(device: str) -> SentenceTransformer:

    model = SentenceTransformer("d0rj/e5-base-en-ru", device=device)

    model.max_seq_length = 512

    model.tokenizer.model_max_length = 512

    return model


@st.cache_resource(show_spinner=False)
def load_reranker(device: str) -> CrossEncoder:

    return CrossEncoder("qilowoq/bge-reranker-v2-m3-en-ru", device=device)


@st.cache_resource(show_spinner=False)
def get_qdrant_client(url: str, api_key: str) -> QdrantClient:

    return QdrantClient(url=url, api_key=api_key, port=443, timeout=30.0)


@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str) -> Groq:

    return Groq(api_key=api_key)


def format_results(items: List[Dict[str, object]]) -> str:

    lines: List[str] = []

    for idx, book in enumerate(items, start=1):

        title = str(book.get("title") or "Без названия")

        author = str(book.get("author") or "Автор неизвестен")

        lines.append(f"{idx}. {title} - {author}")

        annotation = (book.get("annotation") or "").strip()

        if annotation:

            lines.append(f"   Аннотация: {annotation}")

        page_url = (book.get("page_url") or "").strip()

        if page_url:

            lines.append(f"   Ссылка: {page_url}")

        search_score = book.get("search_score")

        rerank_score = book.get("rerank_score")

        score_bits: List[str] = []

        if search_score is not None:

            score_bits.append(f"счёт поиска: {search_score}")

        if rerank_score is not None:

            score_bits.append(f"счёт реранкера: {float(rerank_score):.4f}")

        if score_bits:

            lines.append("   " + ", ".join(score_bits))

    return "\n".join(lines)


def search_books(
    client: QdrantClient,
    embed_model: SentenceTransformer,
    reranker: CrossEncoder,
    query: str,
    collection: str,
    top_k: int,
    fetch_limit: int,
) -> List[Dict[str, object]]:

    query = query.strip()

    if not query:

        return []

    query_vector = embed_model.encode(
        f"query: {query}",
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    hits = client.search(
        collection_name=collection,
        query_vector=query_vector.tolist(),
        limit=fetch_limit,
    )

    if not hits:

        return []

    pairs = [[query, (hit.payload or {}).get("annotation", "")] for hit in hits]

    rerank_scores = reranker.predict(pairs)

    results: List[Dict[str, object]] = []

    for hit, rerank_score in zip(hits, rerank_scores):

        payload = hit.payload or {}

        result = {
            "uuid": payload.get("uuid", ""),
            "title": payload.get("title", "Без названия"),
            "author": payload.get("author", "Автор неизвестен"),
            "annotation": payload.get("annotation", ""),
            "page_url": payload.get("page_url", ""),
            "image_url": payload.get("image_url", ""),
            "search_score": (
                round(float(hit.score), 4) if hit.score is not None else None
            ),
            "rerank_score": float(rerank_score),
        }

        results.append(result)

    results.sort(key=lambda item: item["rerank_score"], reverse=True)

    return results[:top_k]


def generate_answer(
    client: Groq,
    persona_prompt: str,
    query: str,
    results: List[Dict[str, object]],
    temperature: float,
    max_tokens: int,
) -> str:

    if not results:

        return "Подходящих книг не найдено, уточните запрос."

    context_text = format_results(results)

    messages = [
        {"role": "system", "content": persona_prompt.strip()},
        {
            "role": "user",
            "content": (
                f"Запрос читателя: {query.strip()}\n" f"Найденные книги: {context_text}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )

    return response.choices[0].message.content.strip()


def render_sources(items: List[Dict[str, object]]) -> None:
    for idx, book in enumerate(items, start=1):
        title = str(book.get("title") or "Без названия")
        author = str(book.get("author") or "Автор неизвестен")
        st.write(f"{idx}. {title} - {author}")
        link = (book.get("page_url") or "").strip()
        if link:
            st.write(link)
        score_bits: List[str] = []
        similarity = book.get("search_score")
        if similarity is not None:
            score_bits.append(f"similarity: {similarity}")
        rerank = book.get("rerank_score")
        if rerank is not None:
            score_bits.append(f"rerank score: {float(rerank):.4f}")
        if score_bits:
            st.caption(", ".join(score_bits))
        annotation = (book.get("annotation") or "").strip()
        if annotation:
            st.caption(annotation)


def main() -> None:

    st.set_page_config(page_title="Find My Book Chat")

    st.title("Чат для поиска книг")
    st.subheader("Можете выбрать разные личности в настройках")

    with st.sidebar:

        st.header("Настройки")

        persona_name = st.selectbox("Личность ассистента", list(PERSONAS.keys()))

        temperature = st.slider("Температура", 0.0, 1.0, 0.5, 0.1)

        top_k = st.slider("Количество книг", 1, 5, 3)

        show_sources = st.checkbox("Показывать найденные книги", value=True)

        if st.button("Очистить чат"):

            st.session_state.pop("history", None)

            rerun_app()

    groq_api_key = read_secret("GROQ_API_KEY").strip()

    qdrant_url = read_secret("QDRANT_URL").strip()

    qdrant_api_key = read_secret("QDRANT_API_KEY").strip()

    collection_name = (
        read_secret("QDRANT_COLLECTION").strip()
        or os.getenv("QDRANT_COLLECTION", "books_by_annotation")
    ).strip()

    if not groq_api_key or not qdrant_url or not qdrant_api_key:
        st.info(
            "Добавьте ключи Groq и Qdrant в secrets.toml или переменные окружения, чтобы начать диалог."
        )
        return

    embedding_model = load_embedding_model(DEVICE)
    reranker = load_reranker(DEVICE)
    qdrant_client = get_qdrant_client(qdrant_url, qdrant_api_key)
    groq_client = get_groq_client(groq_api_key)
    persona_prompt = PERSONAS[persona_name]
    assistant_avatar = PERSONA_AVATARS.get(persona_name, DEFAULT_ASSISTANT_AVATAR)
    fetch_limit = max(top_k * 4, 12)

    if "history" not in st.session_state:
        st.session_state.history = []

    for message in st.session_state.history:
        role = message["role"]
        stored_avatar = message.get("avatar")
        persona = message.get("persona")
        avatar = stored_avatar or (
            PERSONA_AVATARS.get(persona, DEFAULT_ASSISTANT_AVATAR)
            if role == "assistant"
            else USER_AVATAR
        )
        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])
            if role == "assistant" and show_sources and message.get("sources"):
                with st.expander("Найденные книги", expanded=False):
                    render_sources(message["sources"])

    prompt = st.chat_input("Расскажите, что бы вы хотели почитать")
    if not prompt:
        return

    st.session_state.history.append({"role": "user", "content": prompt, "avatar": USER_AVATAR})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    assistant_reply = ""
    retrieved: List[Dict[str, object]] = []

    with st.chat_message("assistant", avatar=assistant_avatar):
        with st.spinner("Собираю рекомендации..."):
            try:
                retrieved = search_books(
                    qdrant_client,
                    embedding_model,
                    reranker,
                    prompt,
                    collection_name.strip(),
                    top_k,
                    fetch_limit,
                )
            except Exception as error:
                assistant_reply = f"Не удалось выполнить поиск: {error}"
                st.error(assistant_reply)

        if not assistant_reply:
            if not retrieved:
                assistant_reply = "Ничего не нашлось. Попробуйте переформулировать запрос."
            else:
                try:
                    assistant_reply = generate_answer(
                        groq_client,
                        persona_prompt,
                        prompt,
                        retrieved,
                        temperature,
                        max_tokens=900,
                    )
                except Exception as error:
                    assistant_reply = f"Не получилось получить ответ от модели: {error}"

        st.markdown(assistant_reply)
        if show_sources and retrieved:
            with st.expander("Найденные книги", expanded=False):
                render_sources(retrieved)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": assistant_reply,
            "sources": retrieved if retrieved else [],
            "persona": persona_name,
            "avatar": assistant_avatar,
        }
    )


if __name__ == "__main__":

    main()
