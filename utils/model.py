from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Shared model instance for all agents
model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")

