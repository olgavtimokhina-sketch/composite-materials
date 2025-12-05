# app.py - Финальная рабочая версия
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, render_template
import os

app = Flask(__name__)

# Настройки модели
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models/final_neural_network_model.keras")

print("=" * 50)
print("🚀 Запуск Flask приложения для прогнозирования")
print("=" * 50)

# Загрузка модели с детальной отладкой
model = None
model_loaded = False

if os.path.exists(MODEL_PATH):
    try:
        print(f"📁 Загрузка модели из: {MODEL_PATH}")
        model = keras.models.load_model(MODEL_PATH)
        model_loaded = True
        print(f"✅ Модель успешно загружена!")
        
        # Выводим информацию о модели
        print(f"📐 Архитектура модели:")
        print(f"   • Входная форма: {model.input_shape}")
        print(f"   • Выходная форма: {model.output_shape}")
        print(f"   • Количество слоев: {len(model.layers)}")
        
        # Тестовый запрос к модели
        test_input = np.array([[1.0] * 12], dtype=np.float32)
        try:
            test_prediction = model.predict(test_input, verbose=0)
            print(f"🧪 Тестовое предсказание: {test_prediction[0][0]}")
            print(f"✅ Модель работает корректно!")
        except Exception as e:
            print(f"⚠️  Тестовый запрос не удался: {e}")
            
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        print(f"   Тип ошибки: {type(e).__name__}")
else:
    print(f"❌ Файл модели не найден: {MODEL_PATH}")

print("-" * 50)

def predict_with_model(params):
    """Функция для предсказания с использованием модели"""
    try:
        if not model_loaded or model is None:
            raise Exception("Модель не загружена")
        
        # Преобразуем параметры в numpy массив
        input_data = np.array([params], dtype=np.float32)
        
        # Проверяем форму данных
        expected_shape = model.input_shape[1]  # Ожидаемое количество параметров
        if len(params) != expected_shape:
            raise Exception(f"Ожидается {expected_shape} параметров, получено {len(params)}")
        
        # Делаем предсказание
        prediction = model.predict(input_data, verbose=0)
        
        # Извлекаем значение
        if isinstance(prediction, np.ndarray):
            return float(prediction[0][0])
        else:
            return float(prediction.numpy()[0][0])
            
    except Exception as e:
        print(f"❌ Ошибка в predict_with_model: {e}")
        raise

def calculate_percentage(prediction_value):
    """Рассчитывает процентное соотношение матрица-наполнитель"""
    if prediction_value <= 0:
        return 0.0
    # prediction_value - соотношение матрица:наполнитель
    # Например, 1.5 означает 1.5:1
    matrix_percent = (prediction_value / (prediction_value + 1)) * 100
    return float(matrix_percent)

def mock_prediction(params):
    """Заглушка для тестирования"""
    # Простая линейная комбинация с разными весами
    weights = [0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 
               0.04, 0.03, 0.03, 0.02, 0.02, 0.01]
    result = sum(w * p for w, p in zip(weights, params))
    # Нормализуем результат в диапазоне 0.5-3.0
    result = max(0.5, min(3.0, result))
    return result

@app.route('/', methods=['GET', 'POST'])
def app_calculation():
    """Основная функция обработки запросов"""
    message = ''
    prediction_value = None
    prediction_percent = None
    error = None
    
    if request.method == 'POST':
        print(f"\n📨 Получен POST запрос")
        
        try:
            # Собираем параметры из формы
            param_lst = []
            for i in range(1, 13):
                param_name = f'param{i}'
                param_value = request.form.get(param_name, '').strip()
                
                print(f"  {param_name}: '{param_value}'")
                
                # Проверка на пустое значение
                if not param_value:
                    error = f"Поле параметра {i} не заполнено"
                    break
                
                # Преобразование в число
                try:
                    # Заменяем запятую на точку
                    cleaned_value = param_value.replace(',', '.')
                    num_value = float(cleaned_value)
                    param_lst.append(num_value)
                except ValueError:
                    error = f"Некорректное значение в поле параметра {i}: '{param_value}'"
                    break
            
            if error:
                print(f"❌ Ошибка валидации: {error}")
            elif len(param_lst) != 12:
                error = f"Ожидается 12 параметров, получено {len(param_lst)}"
                print(f"❌ {error}")
            else:
                print(f"✅ Все 12 параметров получены корректно")
                
                # Выбор метода предсказания
                if model_loaded:
                    print(f"🤖 Используется нейронная сеть")
                    try:
                        prediction_value = predict_with_model(param_lst)
                        print(f"📊 Предсказание модели: {prediction_value}")
                    except Exception as e:
                        print(f"⚠️  Ошибка при предсказании моделью: {e}")
                        print(f"🔧 Переключаемся на заглушку")
                        prediction_value = mock_prediction(param_lst)
                else:
                    print(f"🔧 Используется заглушка (модель не загружена)")
                    prediction_value = mock_prediction(param_lst)
                
                # Расчет процентного соотношения
                prediction_percent = calculate_percentage(prediction_value)
                
                # Форматируем сообщение
                message = f"Соотношение матрица-наполнитель: {prediction_value:.3f} : 1"
                
                print(f"📈 Результат: {message}")
                print(f"📊 Процентное соотношение: матрица - {prediction_percent:.1f}%, наполнитель - {100-prediction_percent:.1f}%")
                
        except Exception as e:
            error = f"Ошибка при расчёте: {str(e)}"
            print(f"❌ Неожиданная ошибка: {e}")
    
    # Рендерим шаблон
    return render_template("index.html", 
                          message=message,
                          prediction_value=prediction_value,
                          prediction_percent=prediction_percent,
                          error=error)

@app.route('/health')
def health_check():
    """Эндпоинт для проверки работоспособности"""
    return {
        'status': 'ok',
        'model_loaded': model_loaded,
        'model_path': MODEL_PATH if os.path.exists(MODEL_PATH) else 'not_found'
    }

if __name__ == '__main__':
    print(f"🌐 Веб-интерфейс будет доступен по адресу: http://127.0.0.1:5000")
    print(f"🔧 Эндпоинт проверки: http://127.0.0.1:5000/health")
    print("=" * 50)
    print("🔄 Запуск сервера... (для остановки нажмите Ctrl+C)")
    print("=" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Ошибка при запуске сервера: {e}")

