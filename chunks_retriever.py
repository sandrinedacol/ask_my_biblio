import pickle

from database_loader import load_vector_store


def store_test_query_chunk_pairs(queries, db_name='biblio', k=5):
    vector_store = load_vector_store(db_name)
    data = {query: vector_store.similarity_search(query, k=k) for query in queries}
    with open(f'./test_query_chunk_pairs.pkl', 'wb') as f:
        pickle.dump(data, f)


def select_relevant_chunks(query, db_name='biblio', k=5):
    with open(f'./test_query_chunk_pairs.pkl', 'rb') as f:
        data = pickle.load(f)
    if query in data:
        results = data[query]
    else:
        vector_store = load_vector_store(db_name)
        results = vector_store.similarity_search(query, k=k)
        data[query] = results
        with open(f'./test_query_chunk_pairs.pkl', 'wb') as f:
            pickle.dump(data, f)
    return results


if __name__ == "__main__":
    queries = [
        "What is Block Point domain-wall speed?",
        "What is a GROUP BY query?",
        "How many types of domain walls can be found in cylindrical nanowires?",
        'Is it possible to decrease magnetostatic interactions while keeking nanowires in organized arrays?',
        "Define multi-armed bandit algorithm.",
        "What material is deposited during ALD?"
    ]
    store_test_query_chunk_pairs(queries)


