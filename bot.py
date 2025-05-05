# 3. Импорт необходимых библиотек
import math
import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from telegram.constants import ParseMode

from telegram import (
    Update, 
    ReplyKeyboardMarkup, 
    InlineKeyboardMarkup, 
    InlineKeyboardButton
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

# 4. Загрузим и подготовим данные
df = pd.read_excel('таблица оценки.xlsx')
df['Дата просмотра'] = pd.to_datetime(df['Дата просмотра'])
first_date = df['Дата просмотра'].min()
df['Дни от первой даты'] = (df['Дата просмотра'] - first_date).dt.days

features = ['Комедия','Драма','Романтика','Г герои','2 герои','Мир','Кринж']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=5, random_state=42)
df['Кластер'] = kmeans.fit_predict(X_scaled)

cluster_means = df.groupby('Кластер')['Средняя оценка'].mean()
ranked_clusters = cluster_means.sort_values(ascending=False).index
rank_mapping = {ranked_clusters[i]: r for i, r in enumerate(['S','A','B','C','D'])}
df['Ранг'] = df['Кластер'].map(rank_mapping)

cosine_sim = cosine_similarity(X_scaled)

def get_recommendations(idx, top_n=3):
    sim = list(enumerate(cosine_sim[idx]))
    sim = sorted(sim, key=lambda x: x[1], reverse=True)[1:1+top_n]
    inds, scores = zip(*sim)
    return df.iloc[list(inds)], list(scores)

# 5. Основная клавиатура с тремя кнопками
keyboard = ReplyKeyboardMarkup(
    [['📋 Список', '🔮 Рекомендация', '📊 Статистика']],
    resize_keyboard=True
)

# 6a. /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "Привет! Я бот-рекомендатор аниме.\n"
        "Нажмите кнопку:\n"
        "📋 — чтобы получить постраничный список аниме\n"
        "🔮 — чтобы получить рекомендации\n"
        "📊 — чтобы узнать полную статистику по одному аниме",
        reply_markup=keyboard
    )

async def send_anime_page(update_or_query, context, page: int, edit=False):
    per_page = 15
    total = len(df)
    pages = math.ceil(total / per_page)
    start = page * per_page
    end = min(start + per_page, total)
    
    # 1) Заголовок
    text = f"<b>📃 Список аниме — страница {page+1}/{pages}</b>\n"
    
    # 2) Строки с номерами, названием, рангом и оценкой
    for i, row in df.iloc[start:end].iterrows():
        text += (
            f"\n<code>{i:>2}</code>  <b>{row['Название']}</b>\n"
            f"   🎖 <b>Ранг:</b> <code>{row['Ранг']}</code>   ⭐️ <b>Оценка:</b> <code>{row['Средняя оценка']:.2f}</code>\n"
        )
    
    # 3) Inline-кнопки для пагинации
    buttons = []
    if page > 0:
        buttons.append(InlineKeyboardButton("⬅️ Назад", callback_data=f"list_{page-1}"))
    if page < pages - 1:
        buttons.append(InlineKeyboardButton("Вперёд ➡️", callback_data=f"list_{page+1}"))
    markup = InlineKeyboardMarkup([buttons]) if buttons else None

    # 4) Отправка или редактирование
    if edit:
        await update_or_query.edit_message_text(
            text,
            reply_markup=markup,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
        await update_or_query.answer()
    else:
        await update_or_query.message.reply_text(
            text,
            reply_markup=markup,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )

async def list_anime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await send_anime_page(update, context, page=0, edit=False)

async def list_pagination_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    page = int(query.data.split('_')[1])
    await send_anime_page(query, context, page=page, edit=True)

# 6c. Кнопка «Рекомендация»
async def recommend_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    context.user_data['awaiting_recommend'] = True
    await update.message.reply_text(
        f"Введите номер аниме (0–{len(df)-1}) для рекомендаций:",
        reply_markup=keyboard
    )

# 6d. Кнопка «Статистика»
async def stats_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    context.user_data['awaiting_stats'] = True
    await update.message.reply_text(
        f"Введите номер аниме (0–{len(df)-1}) для получения полной статистики:",
        reply_markup=keyboard
    )

# 6e. Обработка ввода числа — рекомендации или статистика
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    # Рекомендации
    if context.user_data.get('awaiting_recommend'):
        idx = int(update.message.text.strip())
        recs, scores = get_recommendations(idx)
        base = df.iloc[idx]['Название']
        
        # 1) Заголовок
        msg = f"<b>🔮 Рекомендации для «<i>{base}</i>»</b>\n"
        
        # 2) Сами рекомендации
        for n, (i, row) in enumerate(recs.iterrows(), start=1):
            genres = [g for g in ['Комедия','Драма','Романтика'] if row[g]==1]
            msg += (
                f"\n<b>{n}. {row['Название']}</b>\n"
                f"   🎖 <b>Ранг:</b> <code>{row['Ранг']}</code>\n"
                f"   ⭐️ <b>Оценка:</b> <code>{row['Средняя оценка']:.2f}</code>\n"
                f"   🔗 <b>Схожесть:</b> <code>{scores[n-1]:.2f}</code>\n"
            )
        await update.message.reply_text(
            msg,
            reply_markup=keyboard,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
        context.user_data.clear()
        return

    # === Полная статистика ===
    if context.user_data.get('awaiting_stats'):
        idx = int(update.message.text.strip())
        row = df.iloc[idx]
        
        # 1) Заголовок
        stat = f"<b>📊 Полная статистика «<i>{row['Название']}</i>»</b>\n"
        
        # 2) Все колонки красиво списком
        for col, val in row.items():
            if isinstance(val, float):
                val = f"{val:.2f}"
            elif pd.isna(val):
                val = "—"
            elif isinstance(val, pd.Timestamp):
                val = val.strftime('%Y-%m-%d')
            stat += f"\n• <b>{col}:</b> <code>{val}</code>"
        
        await update.message.reply_text(
            stat,
            reply_markup=keyboard,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
        context.user_data.clear()
        return

    # По умолчанию
    await update.message.reply_text("Нажмите кнопку для действия.", reply_markup=keyboard)

# 7. Сборка и запуск бота
BOT_TOKEN = "8152150712:AAHbtVMMjoh-3SyjPQTUVE8BYHYF0oBTioE"
app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.Regex('^📋 Список$'), list_anime))
app.add_handler(CallbackQueryHandler(list_pagination_handler, pattern=r"^list_\d+$"))
app.add_handler(MessageHandler(filters.Regex('^🔮 Рекомендация$'), recommend_button))
app.add_handler(MessageHandler(filters.Regex('^📊 Статистика$'), stats_button))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

print("Бот запущен. Ожидаю сообщений…")
app.run_polling()
