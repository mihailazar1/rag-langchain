import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify

from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

from cognitive_search import get_similar_content, get_specific_index_ids, delete_index_chunks_by_id

def create_app():
    application = Flask(__name__)

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                      model_kwargs={"device": "cuda"})


    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=os.environ.get('AZURE_COGNITIVE_SEARCH_ADDRESS'),
        azure_search_key=os.environ.get('AZURE_COGNITIVE_SEARCH_API_KEY'),
        index_name=os.environ.get('AZURE_COGNITIVE_SEARCH_INDEX_NAME'),
        embedding_function=instructor_embeddings.embed_query
    )    

    @application.route("/api/query_specific", methods=["GET"])
    def query_specific():
        input_query = request.args.get('query')
        source_file = request.args.get('document')
        
        docs = get_similar_content(input_query, vector_store)

        response = {'content': {}, 'metadata': {}}

        for i, doc in enumerate(docs):
            if doc.metadata.get('source') == source_file:
                response['content'][f'response {i+1}:'] = doc.page_content
                response['metadata'][f'source {i+1}:'] = doc.metadata


        return jsonify(response)
    

    
    @application.route("/api/query", methods=["GET"])
    def query():
        input_query = request.args.get('query')
        
        docs = get_similar_content(input_query, vector_store)

        response = {'content': {}, 'metadata': {}}

        for i, doc in enumerate(docs):
            response['content'][f'response {i+1}:'] = doc.page_content
            response['metadata'][f'source {i+1}:'] = doc.metadata


        return jsonify(response)
    


    @application.route("/api/delete_specific", methods=["GET"])
    def delete_specific():  
        try:
        
            source_document = request.args.get('document')
            
            ids = get_specific_index_ids(source_document)
            delete_index_chunks_by_id(ids)


            response = {'result': 'Successfully deleted all entries from document'}

            return jsonify(response)

        except Exception as e:
            response = {'result': f'Error: {str(e)}'}
            return jsonify(response)
        
    return application




if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0", port=80, threaded=True)