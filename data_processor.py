import pandas as pd
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['year', 'revenue', 'cogs', 'operating_expenses', 'net_income']

def load_and_process_data(file_path: str) -> pd.DataFrame:
    """
    Загружает финансовые данные из CSV и рассчитывает производные метрики.
    """
    try:
        # Загрузка данных
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='utf-16')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='windows-1251')
        logger.info(f"Успешно загружен файл: {file_path}")
        
        # Валидация структуры
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {', '.join(missing_columns)}")
            
        # Сортировка по году для корректного расчета роста
        df = df.sort_values(by='year').reset_index(drop=True)
        
        # Расчет производных метрик
        
        # Revenue Growth (%): (revenue[t] - revenue[t-1]) / revenue[t-1] * 100
        df['revenue_growth_pct'] = df['revenue'].pct_change() * 100
        
        # Operating Margin (%): (revenue - cogs - operating_expenses) / revenue * 100
        df['operating_margin_pct'] = (df['revenue'] - df['cogs'] - df['operating_expenses']) / df['revenue'] * 100
        
        # Net Margin (%): net_income / revenue * 100
        df['net_margin_pct'] = (df['net_income'] / df['revenue']) * 100
        
        # Округление до 2 знаков после запятой
        df['revenue_growth_pct'] = df['revenue_growth_pct'].round(2)
        df['operating_margin_pct'] = df['operating_margin_pct'].round(2)
        df['net_margin_pct'] = df['net_margin_pct'].round(2)
        
        logger.info("Метрики успешно рассчитаны.")
        return df

    except FileNotFoundError:
        logger.error(f"Файл не найден: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Файл пуст: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при обработке данных: {str(e)}")
        raise

if __name__ == "__main__":
    # Тестирование на базовом датасете
    try:
        processed_df = load_and_process_data("financial_data.csv")
        print("\nПервые 5 строк обработанных данных:")
        print(processed_df.head())
        print("\nИнформация о датафрейме:")
        print(processed_df.info())
    except Exception as e:
        print(f"Тестирование завершилось с ошибкой: {e}")
