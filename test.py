from os import getenv

from dotenv import load_dotenv

load_dotenv()

print(getenv("CORS_ORIGINS", "*").split(","))
