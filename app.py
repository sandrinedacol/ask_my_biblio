import chainlit as cl
from query_answerer import AgenticAnswerer, BasicAnswerer
from dotenv import load_dotenv
import os
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise Exception('Set your Google API key in a .env file')
# Generate or retrieve yout API key:
# https://aistudio.google.com/app/api-keys?hl=fr
# see
# https://ai.google.dev/gemini-api/docs/api-key

from phoenix.otel import register

tracer_provider = register(
    project_name="my-llm-app",
    auto_instrument=True,
)
# lancer uv run phoenix serve pour monitorer le truc en local

tracer = tracer_provider.get_tracer(__name__)

# @cl.on_chat_start
# async def on_chat_start():
#     answerer = AgenticAnswerer()

llm = BasicAnswerer()

@cl.on_message
async def main(message: cl.Message):
    
    answer = llm.answer(message.content)
    await cl.Message(
        content=answer,
    ).send()