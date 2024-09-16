import logging
from langchain.vectorstores import Chroma
from telegram import ForceReply, Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, ConversationHandler, filters
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI



logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma_db"
systempromt = "Тебя зовут атзири , ты  умный  ассистент , который отвечает на вопросы  пользователей."
model_name = "intfloat/multilingual-e5-base"
emb = HuggingFaceEmbeddings(model_name=model_name)


# Initialize the OpenAI client
client = OpenAI(base_url="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", api_key="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

# Function to handle file upload
async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the file upload."""
    global  emb 
    user = update.effective_user
    
    # Check if the user has sent a document
    if update.message.document:  
        file_id = update.message.document.file_id
        file = await context.bot.get_file(file_id)
        
       
        loader = PyPDFLoader(file.file_path)
        documents = loader.load()

        # Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1977, chunk_overlap=61)
        chunks = text_splitter.split_documents(documents)

        # Create a unique directory for the user's Chroma database
        user_chroma_path = f"chroma_db_{user.id}"  # Use user ID to make it unique

        # Delete existing Chroma data if it exists


        # Create a new Chroma database
        chroma = Chroma.from_documents(chunks, emb, persist_directory=user_chroma_path , collection_name='langchain')
            
        await update.message.reply_text(f"Файл  был  успешно  загружен , теперь я  могу  отвечать на вопросы по  нему\n Используйте  команду  /ask  чтобы  задать  вопрос")
        
        # Store Chroma database path in user data
        context.user_data["chroma_path"] = user_chroma_path
    else:
        await update.message.reply_text("Пожалуйста  предоставьте  файл  pdf.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!. Я  языковая  модель которая может  отвечать на твои  вопросы ",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("мои  команды  /clear очищает  все файлы , которые  загрузил пользователь  , чтобы уулчшить поиск  /statr выводит  начлаьное  сообщение ")



async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles user questions using the loaded document."""

    user = update.effective_user
    user_question = update.message.text
    logger.info("Question from User: %s", user_question)

    # Check if Chroma store exists for the user
    if "chroma_path" in context.user_data:
        chroma_path = context.user_data["chroma_path"] 
        chroma = Chroma(persist_directory=chroma_path, embedding_function=emb)
        docs_chroma = chroma.similarity_search_with_score(user_question, k=3)

        context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

        PROMPT_TEMPLATE = '''Используй только следующий контекст, чтобы очень кратко ответить на вопрос в конце.
            Не пытайся выдумывать ответ.
            Контекст:
            ===========
            {context}
            ===========
            Вопрос:
            ===========
            {question}'''
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question= user_question)

        # Generate a response using OpenAI
        completion = client.chat.completions.create(
            model="bartowski/gemma-2-9b-it-GGUF",
            messages=[
                {"role": "system", "content": systempromt},
                {"role": "user", "content":  prompt},
            ],
            temperature=0.7,
        )
        reply = completion.choices[0].message.content
        await update.message.reply_text(reply)
    else:
        await update.message.reply_text("Пожалуйста, сначала загрузите документ, чтобы я мог на него ссылаться.")


 
async def bot_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Returns the reply to user after getting reply from server."""
    user = update.message.from_user
    
    logger.info("Question from User: %s", update.message.text)
    
    if update.message.text != '':
        user_input = update.message.text

        if "chat_history" not in context.user_data:
            context.user_data["chat_history"] = []

        chat_history = context.user_data["chat_history"]
        chat_history.append({"role": "user", "content": user_input})

        completion = client.chat.completions.create(
            model="bartowski/gemma-2-9b-it-GGUF",
            messages=[
                {"role": "system", "content": systempromt},
                *chat_history,
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
        )
        reply = completion.choices[0].message.content
        chat_history.append({"role": "assistant", "content": reply})

        await update.message.reply_text(reply)
    else:
        logger.error("Received empty user input")
        await update.message.reply_text("Пожалуйста, задайте вопрос.")
async def clear_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clears all data for the user."""
    user = update.effective_user

    if "chroma_path" in context.user_data:
        chroma_path = context.user_data["chroma_path"]
        chroma = Chroma(persist_directory=chroma_path, embedding_function=emb)
        chroma.delete_collection()

        

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(
        "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    ).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("ask", ask_question)) 
    application.add_handler(CommandHandler("clear", clear_data)) # Add the /ask command handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_reply))

    application.add_handler(MessageHandler(filters.Document.PDF, handle_file))
  

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()