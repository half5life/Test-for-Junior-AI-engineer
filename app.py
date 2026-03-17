import streamlit as st
import pandas as pd
import os
from data_processor import load_and_process_data
from ai_agent import get_ai_agent

st.set_page_config(page_title="Financial AI Assistant", page_icon="📈", layout="wide")

st.title("📈 AI-ассистент: Финансовый аналитик")
st.markdown("""
Этот прототип AI-ассистента помогает анализировать финансовые данные и отвечать на вопросы о результатах деятельности компании.
По умолчанию загружены данные из `financial_data.csv`, но вы можете загрузить свой собственный файл.
""")

# Инициализация session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "df" not in st.session_state:
    st.session_state.df = None

if "agent" not in st.session_state:
    st.session_state.agent = None

# Sidebar для загрузки файла
with st.sidebar:
    st.header("Настройки данных")
    uploaded_file = st.file_uploader("Загрузить CSV файл", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Сохраняем временно файл
            temp_path = "temp_uploaded.csv"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Если данные обновились, сбрасываем агента
            new_df = load_and_process_data(temp_path)
            if st.session_state.df is None or not st.session_state.df.equals(new_df):
                st.session_state.df = new_df
                st.session_state.agent = None # Нужно переинициализировать агента с новыми данными
                st.session_state.messages = [] # Очищаем историю чата при смене данных
            
            st.success("Данные успешно загружены и обработаны!")
            os.remove(temp_path)
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")
    else:
        if st.session_state.df is None:
            # Загружаем дефолтный файл
            try:
                st.session_state.df = load_and_process_data("financial_data.csv")
                st.success("Загружены данные по умолчанию (financial_data.csv)")
            except Exception as e:
                st.error(f"Не удалось загрузить данные по умолчанию: {e}")

# Инициализация агента при наличии данных
if st.session_state.df is not None:
    # Отображение данных
    with st.expander("📊 Превью данных и рассчитанных метрик", expanded=False):
        st.dataframe(st.session_state.df)

    if st.session_state.agent is None:
        try:
            # Используем модель по умолчанию, заданную в .env или fallback
            model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
            st.session_state.agent = get_ai_agent(st.session_state.df, model_name=model_name)
        except Exception as e:
            st.error(f"Ошибка при инициализации AI-агента: {e}")
            st.info("Убедитесь, что в файле .env указан валидный OPENROUTER_API_KEY")

# Интерфейс чата
st.subheader("Чат с финансовым аналитиком")

# Отображение истории сообщений
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ввод нового сообщения
if prompt := st.chat_input("Задайте вопрос о финансовых показателях (например: В каком году был самый быстрый рост выручки?)"):
    if st.session_state.agent is None:
        st.warning("AI-ассистент не инициализирован. Проверьте настройки API и загрузку данных.")
    else:
        # Добавляем сообщение пользователя в историю
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Получаем ответ от агента
        with st.chat_message("assistant"):
            with st.spinner("Анализирую данные..."):
                try:
                    # Используем invoke для работы с агентом
                    response = st.session_state.agent.invoke(prompt)
                    answer = response.get("output", "Не удалось получить ответ.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Произошла ошибка при обработке запроса: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
