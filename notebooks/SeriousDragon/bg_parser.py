# bg_parser.py
# Требует: requests, beautifulsoup4, lxml

import json
import re
import csv
import os
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4 import NavigableString
from urllib.parse import urljoin

BOOK_KEYS = {"isbn", "издательство", "год издания", "серия", "автор", "кол-во страниц", "страниц"}
AUDIO_WORDS = {
    "аудиокнига", "аудио-книга", "аудио книга", "аудиозапись",
    "mp3", "cd", "audio cd", "аудио cd", "аудио-cd",
    "время звучания", "продолжительность звучания", "битрейт",
    "носитель", "формат аудио", "чтец", "читает", "исполнитель",
    "звучит", "аудио"
}
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")

def clean(t: str, lower: bool = False) -> str:
    s = " ".join((t or "").replace("\xa0", " ").split()).strip()
    return s.lower() if lower else s

def is_book(soup: BeautifulSoup) -> bool:
    """
    Возвращает True, если карточка очень похоже на книгу.
    Логика стала мягче: достаточно ОДНОГО сильного признака (раньше требовали >=2),
    плюс больше источников: JSON-LD, хлебные крошки, блок авторов, таблица характеристик.
    """

    # 1) JSON-LD: @type == Book ИЛИ явно книжные поля внутри объектов
    for tag in soup.find_all("script", type="application/ld+json"):
        txt = tag.string or tag.get_text() or ""
        try:
            data = json.loads(txt)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for obj in items:
            if not isinstance(obj, dict):
                continue
            t = clean(str(obj.get("@type", "")), lower=True)
            if "book" in t:
                return True
            # иногда кладут тип Product, но внутри есть книжные поля
            fields = " ".join([str(obj.get(k, "")) for k in ("isbn", "bookFormat", "author", "publisher", "inLanguage")])
            if any(w in clean(fields, lower=True) for w in ("isbn", "book", "книга", "author", "изд")):
                return True

    # 2) Хлебные крошки: ищем слово "книг" (книги/книга) в разных возможных контейнерах
    bc = soup.select_one(".breadcrumb, nav.breadcrumbs, .breadcrumbs, .bg-breadcrumbs")
    if bc and "книг" in clean(bc.get_text(" "), lower=True):
        return True

    # 3) Блок с авторами, который есть на книгах
    a = soup.select_one("p.goToDescription a.scroll-to") or soup.select_one("p.goToDescription a")
    if a and clean(a.get_text(" "), lower=True):
        return True

    # 4) «Характеристики»: ищем книжные поля хотя бы одно (а не два, как раньше)
    box = soup.find("div", id="collapseExample")
    if box:
        table = box.find("table")
        if table:
            BOOK_KEYS = {"isbn", "издательство", "год издания", "серия", "страниц", "тираж"}
            for cell in table.find_all(["td", "th", "b", "strong"]):
                key = clean(cell.get_text(" "), lower=True).split(":")[0]
                if any(k in key for k in BOOK_KEYS):
                    return True

    # 5) Доп. эвристики по всему тексту страницы (как запасной вариант)
    page_text = clean(soup.get_text(" "), lower=True)
    if ("isbn" in page_text) or ("год издания" in page_text and "страниц" in page_text):
        return True

    return False
# <<< изменено

def is_audiobook(soup: BeautifulSoup) -> bool:
    """True, если карточка — аудиокнига."""
    # 1) JSON-LD @type
    for tag in soup.find_all("script", type="application/ld+json"):
        txt = tag.string or tag.get_text() or ""
        try:
            data = json.loads(txt)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for obj in items:
            if isinstance(obj, dict):
                t = clean(str(obj.get("@type","")), lower=True)
                if "audiobook" in t:       # явный тип
                    return True
                # иногда аудио-признаки лежат в полях description/name
                if any(w in clean(str(obj.get(k,"")), lower=True) for k in ("name","description") for w in AUDIO_WORDS):
                    return True

    # 2) Заголовок и аннотация
    title_el = soup.find("h1")
    if title_el and any(w in clean(title_el.get_text(), lower=True) for w in AUDIO_WORDS):
        return True

    # Аннотация: возьмём блок до «Характеристики» (как в твоей функции)
    desc_box = soup.find("div", id="collapseExample")
    if desc_box and any(w in clean(desc_box.get_text(" "), lower=True) for w in AUDIO_WORDS):
        return True

    # 3) «Характеристики»: ищем аудио-поля
    table = desc_box.find("table") if desc_box else None
    if table:
        for cell in table.find_all(["td","th","b","strong"]):
            txt = clean(cell.get_text(" "), lower=True)
            if any(w in txt for w in AUDIO_WORDS):
                return True

    return False

def extract_title(soup: BeautifulSoup) -> str:
    """
    Берем названия из модуля h1
    """
    h1 = soup.find("h1")
    return clean(h1.get_text()) if h1 else np.nan

def extract_author_from_goToDescription(soup: BeautifulSoup) -> str:
    """
    Берём авторов отсюда:
    <p class="goToDescription"><a class="scroll-to">..., ...</a></p>
    """
    a = soup.select_one("p.goToDescription a.scroll-to") or soup.select_one("p.goToDescription a")
    return clean(a.get_text(" ")) if a else np.nan

def extract_annotation(soup: BeautifulSoup) -> str:
    """
    Аннотация находится в <div id="collapseExample">,перед первым заголовком 'Характеристики'.
    Собираем ВСЕ предыдущие узлы до <h2>/<h3> с таким заголовком.
    """
    box = soup.find("div", id="collapseExample")
    if not box:
        return np.nan

    # Ищем первый заголовок Характеристики (любой регистр/пробелы)
    heading = box.find(
        ["h2", "h3"],
        string=lambda s: s and "характеристик" in s.lower()
    )

    chunks = []
    if heading:
        # Берём все СИБЛИНГИ перед заголовком, в правильном порядке
        prev_nodes = list(heading.previous_siblings)
        prev_nodes.reverse()

    for node in prev_nodes:
        # текстовый узел напрямую
        if isinstance(node, NavigableString):
            text = str(node).strip()
            if text:
                chunks.append(text)
            continue

        # теги, которые точно не нужны
        name = getattr(node, "name", None)
        if name in {"script", "style", "noscript"}:
            continue
        if name == "span" and "overlay_bg" in (node.get("class") or []):
            continue

        # для всех остальных тегов берём .get_text()
        if hasattr(node, "get_text"):
            text = node.get_text(separator="\n", strip=True)
            if text:
                chunks.append(text)

            # вытаскиваем текст узла
            text = node.get_text(separator="\n", strip=True) if hasattr(node, "get_text") else str(node).strip()
            if text:
                chunks.append(text)
    else:
        # На всякий случай: если заголовка нет, берём весь текст блока
        chunks.append(box.get_text(separator="\n", strip=True))

    # Приводим к одной строке
    raw = "\n".join(chunks)
    lines = [clean(line) for line in raw.splitlines() if clean(line)]
    return " ".join(lines)


def extract_image_url(soup: BeautifulSoup, page_url: str) -> str:
    # приоритет: CDN (bgshop.ru/static), иначе og:image
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue
        low = src.lower()
        if "bgshop.ru" in low or "static" in low:
            return urljoin(page_url, src)
    meta = soup.find("meta", {"property": "og:image"}) or soup.find("meta", {"name": "og:image"})
    if meta and meta.get("content"):
        return urljoin(page_url, meta["content"])
    return np.nan

def extract_product(url: str) -> dict:
    """
    Возвращает словарь нужного вида с информацией о книге
    """
    soup = get_soup(url)

    # --- фильтр: пропускаем канцтовары и прочее ---
    if not is_book(soup) or is_audiobook(soup):
        return None  # не книга -> игнор
    
    return {
        "page_url": url,
        "image_url": extract_image_url(soup, url),
        "author": extract_author_from_goToDescription(soup),  
        "title": extract_title(soup),
        "annotation": extract_annotation(soup),
    }

def save_csv(rows, path="books.csv", append=False):
    """
    rows: dict (одна книга) или list[dict] (несколько книг).\n
    append=False -> перезаписать файл\n
    append=True  -> добавить строки
    """
    # приведение к списку словарей
    if isinstance(rows, dict):
        rows = [rows]

    df_new = pd.DataFrame(rows)

    if append and os.path.exists(path):
        # добавляем строки в существующий CSV
        df_new.to_csv(path, mode="a", header=False, index=False, encoding="utf-8")
    else:
        # создаём/перезаписываем файл
        df_new.to_csv(path, index=False, encoding="utf-8")


