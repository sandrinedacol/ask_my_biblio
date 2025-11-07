from langchain_google_genai import ChatGoogleGenerativeAI
from chunks_retriever import select_relevant_chunks

from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent

from dotenv import load_dotenv
import os
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    raise Exception('Set your Google API key in your .env')
# Generate or retrieve yout API key:
# https://aistudio.google.com/app/api-keys?hl=fr
# see
# https://ai.google.dev/gemini-api/docs/api-key


class Answerer():

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key=self.get_api_key()
        )
    
    def get_api_key(self):
        keys_dir = 'api_keys'
        file = f'{keys_dir}/google.txt'
        if os.path.isfile(file):
            with open(file, 'r') as f:
                api_key = f.read()
        else:
            if not os.path.exists(keys_dir):
                os.makedirs(keys_dir)
                with open('.gitignore', 'r') as f:
                    gitignore_content = f.read()
                gitignore_content = f'{keys_dir}/\n{gitignore_content}'
                with open('.gitignore', 'w') as f:
                    f.write(gitignore_content)
            api_key = getpass.getpass("Enter API key for Google Gemini: ")
            with open(file, 'w') as f:
                f.write(api_key)
        return api_key


class BasicAnswerer(Answerer):

    def answer(self, query):
        retrieved_docs = select_relevant_chunks(query)
        docs_content = "\n\n".join(f"file {doc.metadata['source']}, page {doc.metadata['page']+1} : {doc.page_content}" for i, doc in enumerate(retrieved_docs))
        system_message = (
            "You are a helpful assistant. Use the following context in your response:"
            f"\n\n{docs_content}"
            "Provide the name of the file(s) and the page(s) where you found the information, using the following formulation:"
            '''"Information found in <file_name> on page <page_number>."'''
        )
        messages = [
            ("system", system_message),
            ('human', query)
        ]
        ai_msg = self.llm.invoke(messages)
        print(ai_msg.text)


@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = select_relevant_chunks(last_query)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )
    return system_message


class AgenticAnswerer(Answerer):

    def answer(self, query):
        agent = create_agent(self.llm, tools=[], middleware=[prompt_with_context])
        for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()



if __name__ == "__main__":
    queries = [
        "What is Block Point domain-wall speed?",
        "What is a GROUP BY query?",
        "How many types of domain walls can be found in cylindrical nanowires?",
        'Is it possible to decrease magnetostatic interactions while keeking nanowires in organized arrays?',
        "Define multi-armed bandit algorithm.",
        "What material is deposited during ALD?"
    ]
    query = queries[3]
    answerer = AgenticAnswerer()
    answerer.answer(query)
