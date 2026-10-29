"""
AI Product Describer — генератор продающих описаний товаров по фото.

Стек: Python, Streamlit, OpenAI-совместимый LLM API (vision).
Запуск: streamlit run app.py
"""

import base64

import streamlit as st
from openai import OpenAI

# ── Настройки ────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Product Describer", page_icon="🛒", layout="centered")

MODEL = "gpt-4o-mini"          # можно заменить на любую vision-модель
BASE_URL = None                # для DeepSeek/GigaChat укажи свой base_url


def get_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL) if BASE_URL else OpenAI()


TONE_PRESETS = {
    "Продающий (маркетплейсы)": (
        "Убедительный тон для карточек Wildberries/Ozon. "
        "Акцент на выгодах покупателя, эмоциональные триггеры, конкретика вместо воды."
    ),
    "Нейтральный (каталог)": (
        "Деловой информационный тон. Только факты и характеристики, без агрессивных продаж."
    ),
    "Короткое описание": (
        "Максимум 2 предложения, только главное. Подходит для превью и листинга."
    ),
}

PROMPT_TEMPLATE = """Ты — копирайтер маркетплейсов.

Вот фото товара. Определи по изображению:
1. Категория товара
2. Видимые характеристики (материал, цвет, размер, комплектация)
3. Главные преимущества для покупателя

Тонус описания: {tone}

Верни результат строго в формате:

**Заголовок:** (до 60 символов, с ключевыми словами)

**Описание:** (продающий текст 3–5 абзацев, живой язык, без канцелярита)

**Характеристики:**
- (список из 4–6 пунктов)

**Призыв к действию:** (одна строка)

Если на фото не товар, а что-то другое — так и напиши, не выдумывай."""


def encode_image(file) -> str:
    return base64.b64encode(file.getvalue()).decode("utf-8")


def generate_description(image_b64: str, tone: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEMPLATE.format(tone=tone)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        max_tokens=800,
    )
    return response_text(response)


def response_text(resp) -> str:
    return resp.choices[0].message.content.strip()


# ── Интерфейс ────────────────────────────────────────────────────────────
st.title("🛒 AI Product Describer")
st.caption("Фото товара → готовое продающее описание через LLM")

tone = st.selectbox(
    "Тон описания",
    ["Продающий (маркетплейсы)", "Нейтральный", "Короткое описание"],
)

photo = st.file_uploader("Загрузите фото товара", type=["jpg", "jpeg", "png"])

if photo is not None:
    st.image(photo, caption="Загруженный товар", width=300)

    if st.button("✨ Сгенерировать описание", type="primary"):
        with st.spinner("Модель изучает фото и пишет описание..."):
            try:
                b64 = encode_image(photo)
                result = generate_description(b64, tone)
                st.subheader("Результат")
                st.markdown(result)
                st.button("📋 Копировать", on_click=lambda: None)  # ручное копирование
            except Exception as e:  # noqa: BLE001
                st.error(f"Ошибка запроса к API: {e}")
else:
    st.info("Загрузите фото товара, чтобы получить описание")
