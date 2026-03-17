import pandas as pd
from data_processor import load_and_process_data
from ai_agent import get_ai_agent
import os
from dotenv import load_dotenv

load_dotenv()

def test_queries():
    data = load_and_process_data("financial_data.csv")
    ai_analyst = get_ai_agent(data, "stepfun/step-3.5-flash:free")
    
    queries = [
        "В каком году была самая высокая чистая маржа (net_margin_pct)?",
        "Сравните выручку 2005 и 2024 года.",
        "Какая средняя выручка за весь период?"
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"Запрос: {query}")
        try:
            response = ai_analyst.invoke(query)
            print(f"\nОтвет:\n{response['output']}")
        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")

if __name__ == "__main__":
    test_queries()
