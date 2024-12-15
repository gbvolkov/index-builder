from dotenv import load_dotenv,dotenv_values
import os

from pathlib import Path

documents_path = Path.home() / ".env"

load_dotenv(os.path.join(documents_path, 'gv.env'))

CHROME_DRIVER_PATH = './bin/chromedriver.exe'
LOGIN_URL = 'https://kb.ileasing.ru/auth/sign-in?redirect=%2Fspace%2Fa100dc8d-3af0-418c-8634-f09f1fdb06f2%2Farticle%2Fa1038bbc-e5d9-4b5a-9482-2739c19cb6cb'
TARGET_URL = 'https://kb.ileasing.ru/space/a100dc8d-3af0-418c-8634-f09f1fdb06f2/article/a1038bbc-e5d9-4b5a-9482-2739c19cb6cb'
USERNAME = '7810155'
PASSWORD = os.environ.get('IL_PWD') 

LOCAL_MODEL_NAME='/models/Meta-Llama-3.1-8B-Instruct-Q8_0'

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_SD_ASSISTANT_BOT_TOKEN')
#EMBEDDING_MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

#EMBEDDING_MODEL='sentence-transformers/distiluse-base-multilingual-cased-v1'
EMBEDDING_MODEL='/models/multilingual-e5-large'
#EMBEDDING_MODEL='jinaai/jina-embeddings-v3'

GIGA_CHAT_USER_ID=os.environ.get('GIGA_CHAT_USER_ID')
GIGA_CHAT_SECRET = os.environ.get('GIGA_CHAT_SECRET')
GIGA_CHAT_AUTH = os.environ.get('GIGA_CHAT_AUTH')
GIGA_CHAT_SCOPE = "GIGACHAT_API_PERS"

LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
#LANGCHAIN_API_KEY_DEV = os.environ.get('LANGCHAIN_API_KEY_DEV')
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

YA_API_KEY = os.environ.get('YA_API_KEY')
YA_FOLDER_ID = os.environ.get('YA_FOLDER_ID')
YA_AUTH_TOKEN = os.environ.get('YA_AUTH_TOKEN')

GEMINI_API_KEY=os.environ.get('GEMINI_API_KEY')

CHECK_RIGHTS=os.environ.get('CHECK_RIGHTS', default='False')

USERS_SHEET_ID=os.environ.get('USERS_SHEET_ID')
FEEDBACK_SHEET_ID=os.environ.get('FEEDBACK_SHEET_ID')
GOOGLE_SHEETS_CRED=os.environ.get('GOOGLE_SHEETS_CRED')


JINA_API_KEY=os.environ.get('JINA_API_KEY')

def reload_admin_config():
    global CHECK_RIGHTS
    env_vars = dotenv_values(os.path.join(documents_path, 'gv.env'))
    if 'CHECK_RIGHTS' in env_vars:
        CHECK_RIGHTS=env_vars['CHECK_RIGHTS']
    else:
        CHECK_RIGHTS='False'
