import html
import streamlit as st

def display_book(title, author, link, img, annotation, score=None):
    # Обрабатываем описание
    desc = str(annotation) if annotation else "Нет описания"
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
    
    # Формируем строку с автором (добавляем score если есть)
    #author_line = f"{author} • Релевантность: {score * 100}%" if score else author
    
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
                    <a href="{link}" target="_blank">
                        <img src="{img}" style="width:100px; border-radius: 6px;"/>
                    </a>
                </td>
                <td style="padding: 5px; vertical-align: top; border: none;">
                    <b><a href="{link}" target="_blank">{title}</a></b><br>
                    <i>{author}</i><br>
                    {desc_preview}
                    {show_details}
                </td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)