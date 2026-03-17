import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv

# Загрузка переменных окружения из .env
load_dotenv()

def get_ai_agent(df: pd.DataFrame, model_name: str = None):
    """
    Создает и возвращает LangChain Pandas DataFrame Agent, настроенный для финансового анализа.
    """
    
    # Получаем настройки из .env или используем значения по умолчанию
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = "https://openrouter.ai/api/v1"
    
    if not model_name:
        model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY не найден в переменных окружения.")

    # Инициализация LLM через OpenRouter
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0,  # Для точности финансовых расчетов
        default_headers={
            "HTTP-Referer": "https://github.com/HR-amocrm/Test-for-Junior-AI-engineer", # Опционально для OpenRouter
            "X-Title": "Financial AI Assistant Prototype", # Опционально для OpenRouter
        }
    )

    # Системный промпт
    prefix = """
    Вы — Старший финансовый аналитик. Ваша задача — анализировать финансовые данные компании 
    и отвечать на вопросы пользователей на основе предоставленной таблицы (dataframe `df`).

    ПРАВИЛА РАБОТЫ:
    1. Используйте ТОЛЬКО данные из предоставленной таблицы.
    2. Если в данных нет ответа на вопрос, честно скажите об этом.
    3. Обязательно объясняйте ход своих расчетов (какие колонки использовали, какие операции проводили).
    4. Ответы должны быть структурированными и профессиональными.
    5. Всегда округляйте финансовые показатели до двух знаков после запятой, если не указано иное.
    
    ДОСТУПНЫЕ МЕТРИКИ (уже рассчитаны в df):
    - `revenue_growth_pct`: Рост выручки год к году (%)
    - `operating_margin_pct`: Операционная маржа (%)
    - `net_margin_pct`: Чистая маржа (%)
    
    Язык ответов: Русский.
    """

    # Создание агента
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type="openai-tools", # Или "zero-shot-react-description" если модель старая
        allow_dangerous_code=True, # Необходимо для выполнения Python кода агентом
        prefix=prefix
    )

    return agent

if __name__ == "__main__":
    # Быстрый тест (загружаем данные через наш процессор)
    from data_processor import load_and_process_data
    
    try:
        data = load_and_process_data("financial_data.csv")
        ai_analyst = get_ai_agent(data)
        
        # Пример простого запроса
        test_query = "В каком году был самый быстрый рост выручки?"
        print(f"\nЗапрос: {test_query}")
        response = ai_analyst.invoke(test_query)
        print(f"\nОтвет:\n{response['output']}")
        
    except Exception as e:
        print(f"Ошибка при тестировании агента: {e}")
