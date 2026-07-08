"""KZ Admission Helper Bot — Telegram bot for university admission in Kazakhstan."""

import os
import io
import json
import logging
import tempfile
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from telegram import Update, BotCommand
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes,
    filters, ConversationHandler
)
from openai import AsyncOpenAI

import rag
from knowledge import SYSTEM_PROMPT, detect_doc_type
from documents import process_document

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Clients
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Onboarding steps
ONBOARD_NAME, ONBOARD_AGE, ONBOARD_INTERESTS, ONBOARD_BUDGET, ONBOARD_DIRECTION = range(5)


# ═══════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════

def build_context(user_id: int, user_message: str) -> str:
    """Build context from RAG memory for the AI."""
    parts = []

    # User profile
    profile = rag.get_user_profile(user_id)
    if profile:
        parts.append(f"ПРОФИЛЬ ЮЗЕРА: Имя: {profile.get('name', '?')}, Возраст: {profile.get('age', '?')}, "
                      f"Город: {profile.get('city', '?')}, Интересы: {profile.get('interests', '?')}, "
                      f"Бюджет: {profile.get('budget', '?')}, Направление: {profile.get('direction', '?')}, "
                      f"Целевой вуз: {profile.get('target_university', '?')}")

    # Search relevant memories
    try:
        memories = rag.search_memories(user_id, user_message, limit=5)
        if memories:
            mem_texts = [f"[{m['category']}] {m['content']}" for m in memories]
            parts.append("РЕЛЕВАНТНЫЕ ВОСПОМИНАНИЯ:\n" + "\n".join(mem_texts))
    except Exception:
        pass

    # Recent memories for continuity
    recent = rag.get_recent_memories(user_id, limit=5)
    if recent:
        rec_texts = [f"[{m['category']}] {m['content']}" for m in recent]
        parts.append("НЕДАВНИЕ СООБЩЕНИЯ:\n" + "\n".join(rec_texts))

    # Document checklist status
    checklist = rag.get_document_checklist(user_id)
    uploaded = [d for d in checklist if d["uploaded"]]
    missing = [d for d in checklist if not d["uploaded"]]
    if uploaded or missing:
        parts.append(f"ДОКУМЕНТЫ: Загружено {len(uploaded)}/{len(checklist)}")
        if uploaded:
            parts.append("Есть: " + ", ".join(d["name"] for d in uploaded))
        if missing:
            parts.append("Нужны: " + ", ".join(d["name"] for d in missing))

    return "\n\n".join(parts)


async def ask_gpt(user_id: int, user_message: str, extra_context: str = "") -> str:
    """Send message to GPT-4o with full context."""
    context = build_context(user_id, user_message)
    if extra_context:
        context += "\n\n" + extra_context

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context},
        {"role": "user", "content": user_message}
    ]

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=4096,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT error: {e}")
        return f"Ошибка при обращении к AI: {e}"


async def transcribe_voice(file_bytes: bytes) -> str:
    """Transcribe voice message using Whisper API."""
    try:
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(file_bytes)
            tmp_path = f.name

        with open(tmp_path, "rb") as audio_file:
            transcript = await openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ru"
            )
        os.unlink(tmp_path)
        return transcript.text
    except Exception as e:
        logger.error(f"Whisper error: {e}")
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        return ""


# ═══════════════════════════════════════════
# COMMAND HANDLERS
# ═══════════════════════════════════════════

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start — welcome + begin onboarding."""
    user_id = update.effective_user.id
    profile = rag.get_user_profile(user_id)

    if profile and profile.get("onboarded"):
        await update.message.reply_text(
            f"С возвращением, {profile.get('name', 'друг')}! 👋\n\n"
            f"Я помню тебя. Чем могу помочь?\n\n"
            f"Используй /help чтобы увидеть все команды."
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "Привет! 👋 Я твой помощник по поступлению в вузы Казахстана.\n\n"
        "Помогу со всем: выбор вуза, документы, переезд, армия — всё расскажу.\n\n"
        "Давай познакомимся! Как тебя зовут?"
    )
    rag.upsert_user_profile(user_id, onboard_step="name")
    return ONBOARD_NAME


async def onboard_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    name = update.message.text.strip()
    rag.upsert_user_profile(user_id, name=name, onboard_step="age")
    rag.store_memory(user_id, f"Юзера зовут {name}", category="profile")
    await update.message.reply_text(f"Приятно познакомиться, {name}! Сколько тебе лет?")
    return ONBOARD_AGE


async def onboard_age(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        age = int(update.message.text.strip())
    except ValueError:
        await update.message.reply_text("Напиши число, пожалуйста 😄")
        return ONBOARD_AGE
    rag.upsert_user_profile(user_id, age=age, onboard_step="interests")
    rag.store_memory(user_id, f"Возраст: {age} лет", category="profile")
    await update.message.reply_text(
        "Отлично! Какие направления тебе интересны?\n\n"
        "Например: IT, программирование, бизнес, инженерия, медицина, право..."
    )
    return ONBOARD_INTERESTS


async def onboard_interests(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    interests = update.message.text.strip()
    rag.upsert_user_profile(user_id, interests=interests, onboard_step="budget")
    rag.store_memory(user_id, f"Интересы: {interests}", category="profile")
    await update.message.reply_text(
        "Супер! Какой примерный бюджет на обучение в год?\n\n"
        "Можешь написать в рублях или тенге. Например:\n"
        "• до 100 000 руб/год\n"
        "• 100-200 тыс руб/год\n"
        "• до 300 тыс руб/год\n"
        "• деньги не проблема"
    )
    return ONBOARD_BUDGET


async def onboard_budget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    budget = update.message.text.strip()
    rag.upsert_user_profile(user_id, budget=budget, onboard_step="direction")
    rag.store_memory(user_id, f"Бюджет: {budget}", category="profile")
    await update.message.reply_text(
        "Понял! Последний вопрос — на каком языке хочешь учиться?\n\n"
        "• Русский\n"
        "• Английский\n"
        "• Казахский\n"
        "• Без разницы"
    )
    return ONBOARD_DIRECTION


async def onboard_direction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    direction = update.message.text.strip()
    rag.upsert_user_profile(user_id, direction=direction, onboarded=1, onboard_step="done")
    rag.store_memory(user_id, f"Язык обучения: {direction}", category="profile")

    profile = rag.get_user_profile(user_id)

    # Generate personalized recommendation
    recommendation = await ask_gpt(
        user_id,
        f"Юзер только что заполнил анкету. Дай краткую персональную рекомендацию: "
        f"какие вузы подойдут, и что делать в первую очередь. "
        f"Профиль: имя={profile.get('name')}, возраст={profile.get('age')}, "
        f"интересы={profile.get('interests')}, бюджет={profile.get('budget')}, "
        f"язык обучения={direction}"
    )

    await update.message.reply_text(
        f"Отлично, {profile.get('name')}! Анкета заполнена ✅\n\n"
        f"{recommendation}\n\n"
        f"📋 Используй /help для списка всех команд\n"
        f"💬 Или просто пиши мне любые вопросы!"
    )
    return ConversationHandler.END


async def onboard_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Окей, можешь вернуться позже через /start")
    return ConversationHandler.END


async def cmd_docs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /docs — document checklist."""
    user_id = update.effective_user.id
    checklist = rag.get_document_checklist(user_id)

    lines = ["📋 **ЧЕКЛИСТ ДОКУМЕНТОВ**\n"]
    uploaded_count = 0
    for doc in checklist:
        if doc["uploaded"]:
            lines.append(f"✅ {doc['name']} — загружен ({doc['date'][:10]})")
            uploaded_count += 1
        else:
            lines.append(f"❌ {doc['name']}")

    lines.append(f"\n📊 Прогресс: {uploaded_count}/{len(checklist)}")
    lines.append("\n💡 Чтобы загрузить документ — просто отправь фото или файл с подписью (например: \"аттестат\")")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_plan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /plan — current action plan with deadlines."""
    user_id = update.effective_user.id
    response = await ask_gpt(
        user_id,
        "Покажи мой текущий пошаговый план действий с дедлайнами. "
        "Учитывай что сейчас март 2026. Отметь что уже сделано (по загруженным документам) "
        "и что нужно сделать на этой неделе. Будь конкретным с датами."
    )
    await update.message.reply_text(response)


async def cmd_universities(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /universities — university comparison."""
    user_id = update.effective_user.id
    response = await ask_gpt(
        user_id,
        "Подбери и сравни вузы для меня на основе моего профиля. "
        "Покажи таблицу/сравнение: название, стоимость, направления, плюсы/минусы, "
        "проходной балл. Дай конкретную рекомендацию какой выбрать и почему."
    )
    await update.message.reply_text(response)


async def cmd_army(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /army — military service info."""
    user_id = update.effective_user.id
    response = await ask_gpt(
        user_id,
        "Расскажи всё про армию для моей ситуации. Учитывай мой возраст. "
        "Пошагово: что делать с военкоматом, когда сниматься с учёта, "
        "какие документы нужны, риски. Будь конкретным."
    )
    await update.message.reply_text(response)


async def cmd_move(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /move — relocation plan."""
    user_id = update.effective_user.id
    profile = rag.get_user_profile(user_id)
    origin = (profile or {}).get("city") or "моего города"
    response = await ask_gpt(
        user_id,
        f"Составь детальный план переезда из {origin} в Алматы для меня. "
        "Пошагово: билеты, что взять с собой, первые дни в Алматы "
        "(ИИН, Kaspi, сим-карта, жильё, регистрация, РВП). "
        "С ценами и конкретными адресами/ссылками."
    )
    await update.message.reply_text(response)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status — overall preparation status."""
    user_id = update.effective_user.id
    profile = rag.get_user_profile(user_id)
    checklist = rag.get_document_checklist(user_id)
    uploaded = [d for d in checklist if d["uploaded"]]

    status_parts = ["📊 **СТАТУС ПОДГОТОВКИ**\n"]

    if profile:
        status_parts.append(f"👤 {profile.get('name', '?')}, {profile.get('age', '?')} лет")
        status_parts.append(f"🎯 Направление: {profile.get('interests', 'не указано')}")
        status_parts.append(f"💰 Бюджет: {profile.get('budget', 'не указан')}")
        status_parts.append(f"🏫 Целевой вуз: {profile.get('target_university', 'не выбран')}")
    else:
        status_parts.append("⚠️ Профиль не заполнен — используй /start")

    status_parts.append(f"\n📋 Документы: {len(uploaded)}/{len(checklist)}")
    status_parts.append(f"📅 Дата: {datetime.now().strftime('%d.%m.%Y')}")

    today = datetime.now()
    deadline = datetime(2026, 6, 15)
    days_left = (deadline - today).days
    if days_left > 0:
        status_parts.append(f"⏰ До начала приёма документов: {days_left} дней")
    else:
        status_parts.append("🔴 Приём документов уже идёт!")

    response = await ask_gpt(user_id, "Дай краткую оценку моего статуса подготовки и что нужно сделать срочно.")
    status_parts.append(f"\n💡 {response}")

    await update.message.reply_text("\n".join(status_parts), parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help — list commands."""
    await update.message.reply_text(
        "🤖 **КОМАНДЫ БОТА**\n\n"
        "/start — Начало работы, заполнение анкеты\n"
        "/docs — Чеклист документов (✅/❌)\n"
        "/plan — Текущий план действий с дедлайнами\n"
        "/universities — Подбор и сравнение вузов\n"
        "/army — Всё про армию и отсрочку\n"
        "/move — План переезда Москва → Алматы\n"
        "/status — Общий статус подготовки\n"
        "/help — Список команд\n\n"
        "💬 Просто пиши любые вопросы — я отвечу!\n"
        "📎 Отправляй документы (фото/PDF/DOCX) — я сохраню\n"
        "🎤 Можешь отправлять голосовые — я пойму",
        parse_mode="Markdown"
    )


# ═══════════════════════════════════════════
# MESSAGE HANDLERS
# ═══════════════════════════════════════════

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular text messages."""
    user_id = update.effective_user.id
    text = update.message.text

    # Save to memory
    rag.store_memory(user_id, f"Юзер: {text}", category="conversation")

    # Get AI response
    response = await ask_gpt(user_id, text)

    # Save response to memory
    rag.store_memory(user_id, f"Бот: {response[:500]}", category="conversation")

    # Split long messages (Telegram limit 4096)
    if len(response) <= 4096:
        await update.message.reply_text(response)
    else:
        for i in range(0, len(response), 4096):
            await update.message.reply_text(response[i:i+4096])


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages — transcribe and respond."""
    user_id = update.effective_user.id

    await update.message.reply_text("🎧 Слушаю голосовое...")

    # Download voice
    voice_file = await update.message.voice.get_file()
    voice_bytes = await voice_file.download_as_bytearray()

    # Transcribe
    text = await transcribe_voice(bytes(voice_bytes))

    if not text:
        await update.message.reply_text("Не удалось распознать голосовое 😕 Попробуй ещё раз или напиши текстом.")
        return

    await update.message.reply_text(f"📝 Распознано: _{text}_", parse_mode="Markdown")

    # Save and respond
    rag.store_memory(user_id, f"Юзер (голосовое): {text}", category="conversation")
    response = await ask_gpt(user_id, text)
    rag.store_memory(user_id, f"Бот: {response[:500]}", category="conversation")

    if len(response) <= 4096:
        await update.message.reply_text(response)
    else:
        for i in range(0, len(response), 4096):
            await update.message.reply_text(response[i:i+4096])


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle document uploads (PDF, DOCX, etc.)."""
    user_id = update.effective_user.id
    doc = update.message.document
    caption = update.message.caption or doc.file_name or ""

    await update.message.reply_text(f"📄 Получил документ: {doc.file_name}. Обрабатываю...")

    # Download
    file = await doc.get_file()
    file_bytes = await file.download_as_bytearray()

    # Extract text
    extracted = await process_document(bytes(file_bytes), doc.file_name, openai_client)

    # Detect doc type
    doc_type = detect_doc_type(caption or doc.file_name or "")

    # Save
    rag.save_document(user_id, doc_type, doc.file_name, extracted)

    response_parts = [f"✅ Документ сохранён: {doc.file_name}"]
    if doc_type != "other":
        response_parts.append(f"📋 Тип: {doc_type}")
    if extracted:
        preview = extracted[:300] + "..." if len(extracted) > 300 else extracted
        response_parts.append(f"📝 Извлечённый текст:\n{preview}")
    else:
        response_parts.append("⚠️ Не удалось извлечь текст автоматически. Но документ сохранён!")

    response_parts.append("\n💡 Если тип определён неверно, напиши какой это документ.")

    await update.message.reply_text("\n".join(response_parts))


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo uploads — extract text via OCR/Vision."""
    user_id = update.effective_user.id
    caption = update.message.caption or ""

    await update.message.reply_text("📸 Получил фото. Обрабатываю...")

    # Download highest quality photo
    photo = update.message.photo[-1]  # largest size
    file = await photo.get_file()
    file_bytes = await file.download_as_bytearray()

    # Extract text
    extracted = await process_document(bytes(file_bytes), "photo.jpg", openai_client)

    # Detect doc type from caption
    doc_type = detect_doc_type(caption) if caption else "other"

    # Save
    file_name = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    rag.save_document(user_id, doc_type, file_name, extracted)

    response_parts = [f"✅ Фото сохранено"]
    if doc_type != "other":
        response_parts.append(f"📋 Тип документа: {doc_type}")
    if extracted:
        preview = extracted[:300] + "..." if len(extracted) > 300 else extracted
        response_parts.append(f"📝 Распознанный текст:\n{preview}")

    if caption:
        # Also respond to caption as a question
        rag.store_memory(user_id, f"Юзер отправил фото с подписью: {caption}", category="document")
        response_parts.append("")
        gpt_response = await ask_gpt(
            user_id,
            f"Юзер отправил фото документа с подписью: '{caption}'. "
            f"Извлечённый текст: {extracted[:500] if extracted else 'не удалось распознать'}. "
            f"Ответь на подпись и прокомментируй документ.",
            extra_context=f"Извлечённый текст из фото: {extracted[:1000]}" if extracted else ""
        )
        response_parts.append(gpt_response)
    else:
        response_parts.append("\n💡 Совет: добавляй подпись к фото (например 'аттестат', 'паспорт') чтобы я правильно определил тип документа.")

    full_response = "\n".join(response_parts)
    if len(full_response) <= 4096:
        await update.message.reply_text(full_response)
    else:
        for i in range(0, len(full_response), 4096):
            await update.message.reply_text(full_response[i:i+4096])


# ═══════════════════════════════════════════
# ERROR HANDLER
# ═══════════════════════════════════════════

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors."""
    logger.error(f"Update {update} caused error: {context.error}")
    if update and update.message:
        await update.message.reply_text(
            "Произошла ошибка 😕 Попробуй ещё раз или напиши /help"
        )


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

async def post_init(application: Application):
    """Set bot commands after initialization."""
    commands = [
        BotCommand("start", "Начало работы"),
        BotCommand("docs", "Чеклист документов"),
        BotCommand("plan", "План действий"),
        BotCommand("universities", "Подбор вузов"),
        BotCommand("army", "Армия и отсрочка"),
        BotCommand("move", "План переезда"),
        BotCommand("status", "Статус подготовки"),
        BotCommand("help", "Список команд"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands set successfully")


def main():
    """Start the bot."""
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env")

    # Initialize RAG DB
    rag.init_db()

    # Build application
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()

    # Onboarding conversation
    onboard_handler = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            ONBOARD_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_name)],
            ONBOARD_AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_age)],
            ONBOARD_INTERESTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_interests)],
            ONBOARD_BUDGET: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_budget)],
            ONBOARD_DIRECTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, onboard_direction)],
        },
        fallbacks=[CommandHandler("cancel", onboard_cancel)],
    )
    app.add_handler(onboard_handler)

    # Command handlers
    app.add_handler(CommandHandler("docs", cmd_docs))
    app.add_handler(CommandHandler("plan", cmd_plan))
    app.add_handler(CommandHandler("universities", cmd_universities))
    app.add_handler(CommandHandler("army", cmd_army))
    app.add_handler(CommandHandler("move", cmd_move))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help", cmd_help))

    # Message handlers
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Error handler
    app.add_error_handler(error_handler)

    # Start polling
    logger.info("Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
