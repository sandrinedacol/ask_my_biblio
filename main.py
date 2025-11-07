import time

from retriever import select_relevant_chunks
from answer_writer import formulate_answer


def answer_one_question(query):
    chunks = select_relevant_chunks(query)
    formulate_answer(chunks)


def main():
    done = False
    while not done:
        user_input = input('Ask something or quit (q):\n')
        if user_input in ['exit', 'quit', 'q', '']:
            done = True
        else:
            query = user_input
            answer_one_question(query)
            

if __name__ == "__main__":
    query = "What is Block Point domain-wall speed?"
    answer_one_question(query)
    # main()
