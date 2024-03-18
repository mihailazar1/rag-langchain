import os
from dotenv import load_dotenv

load_dotenv()

from blob_manager import get_files
from langchain_community.vectorstores.azuresearch import AzureSearch

from InstructorEmbedding import INSTRUCTOR
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

from cognitive_search import store_embeddings, get_all_index_ids, delete_index_chunks_by_id, get_similar_content



instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                      model_kwargs={"device": "cuda"})




vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=os.environ.get('AZURE_COGNITIVE_SEARCH_ADDRESS'),
    azure_search_key=os.environ.get('AZURE_COGNITIVE_SEARCH_API_KEY'),
    index_name=os.environ.get('AZURE_COGNITIVE_SEARCH_INDEX_NAME'),
    embedding_function=instructor_embeddings.embed_query
)


# if __name__ == '__main__':
    # ids = get_all_index_ids()
    # delete_index_chunks_by_id(ids)

    # store_embeddings(vector_store, "someprefix")
    
    
    # docs = get_similar_content("faith commands us to accept the matter without doubt respecting the wisdom of the arrangement", vector_store)
    
    # for doc in docs:
    #     print(doc.page_content)

