import chainlit as cl
from query_answerer import AgenticAnswerer, BasicAnswerer

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