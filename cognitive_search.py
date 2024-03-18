import os


import blob_manager
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.azuresearch import AzureSearch


from dotenv import load_dotenv
load_dotenv()

def store_embeddings(vector_store: AzureSearch, prefix: str):
    docs = blob_manager.get_files(prefix)

    for doc in docs:
        doc.metadata['source'] = doc.metadata.get('source').split("/")[-1]
        print(doc.metadata.get('source'))



    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, chunk_overlap=150, length_function=len
    )

    split_docs = text_splitter.split_documents(docs)

    for split_doc in split_docs:
        split_doc.metadata['source'] = split_doc.metadata.get('source').split("/")[-1]

    vector_store.add_documents(documents=split_docs)

    print("Data loaded successfully\n")


def get_all_index_ids():
    credentials = AzureKeyCredential(os.environ.get('AZURE_COGNITIVE_SEARCH_API_KEY'))
    search_client = SearchClient(os.environ.get('AZURE_COGNITIVE_SEARCH_ADDRESS'), os.environ.get('AZURE_COGNITIVE_SEARCH_INDEX_NAME'), credentials)

    results = search_client.search(search_text='*')

    ids = []
    for result in results:
        ids.append(result['id'] )

    
    return ids


def get_specific_index_ids(source: str):
    credentials = AzureKeyCredential(os.environ.get('AZURE_COGNITIVE_SEARCH_API_KEY'))
    search_client = SearchClient(os.environ.get('AZURE_COGNITIVE_SEARCH_ADDRESS'), os.environ.get('AZURE_COGNITIVE_SEARCH_INDEX_NAME'), credentials)

    results = search_client.search(search_text='*')

    ids = []
    for result in results:
        if(source in result['metadata']):
            ids.append(result['id'] )

    
    return ids


def delete_index_chunks_by_id(id_list):
    try:
        credentials = AzureKeyCredential(os.environ.get('AZURE_COGNITIVE_SEARCH_API_KEY'))
        search_client = SearchClient(os.environ.get('AZURE_COGNITIVE_SEARCH_ADDRESS'), os.environ.get('AZURE_COGNITIVE_SEARCH_INDEX_NAME'), credentials)

        id_dictionary = []
        for chunk_id in id_list:
            id_dictionary.append({'id':chunk_id})

        search_client.delete_documents(id_dictionary)
        print(f"Deleted {len(id_list)} IDs")
    
    except Exception as e:
        print(f"Error: {str(e)}")





def get_similar_content(user_input: str, vector_store: AzureSearch):
    docs: list[Document] = vector_store.similarity_search(
        query=user_input,
        k=3,
        search_type="similarity",
    )

    return docs