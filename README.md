# Telegram-бот-рекомендатор

## Локальный запуск

1. Склонировать репозиторий  
2. `python -m venv venv && source venv/bin/activate`  
3. `pip install -r requirements.txt`  
4. Скопировать `.env.example` в `.env` и заполнить BOT_TOKEN  
5. `python bot.py`

## Деплой на Railway

1. Завести новый проект на https://railway.app  
2. Подключить GitHub-репозиторий  
3. В разделе Settings → Environment Variables добавить:
   - `BOT_TOKEN` — токен вашего бота  
4. Залить файл `таблица_оценки.xlsx` в корень репозитория  
5. Нажать «Deploy» — и бот автоматически запустится!
