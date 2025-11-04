from langsmith import Client
import os
from langchain_core.prompts import ChatPromptTemplate
from agents.router_agent import router_system_prompt
from dotenv import load_dotenv
load_dotenv()

ls_client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))

prompt_template = ChatPromptTemplate.from_template(router_system_prompt)
ls_client.push_prompt("router_system_prompt", object=prompt_template)