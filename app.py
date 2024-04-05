import os
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify, make_response

from langchain.vectorstores.azuresearch import AzureSearch
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

from cognitive_search import store_embeddings, get_similar_content, get_specific_index_ids, delete_index_chunks_by_id, get_all_index_ids

# For the LLM
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

from model import get_model_response
from flask_sqlalchemy import SQLAlchemy
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps
import secrets

secret = secrets.token_hex(16)




application = Flask(__name__)

application.config['SECRET_KEY'] = 'thisissecret' # = secret
application.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ragdb.sqlite3'
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(application)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True)
    name = db.Column(db.String(50))
    password = db.Column(db.String(80))
    admin = db.Column(db.Boolean)



def token_required(f): # f is a function that is decorated
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.filter_by(public_id=data['public_id']).first()
        except: 
            return jsonify({'message': 'Token is invalid!'}), 401
        
        return f(current_user, *args, **kwargs) # pass user to the route
    
    return decorated # return the decorated function inside of the token required decorator 

def create_app():
    
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                      model_kwargs={"device": "cuda"})


    application.vector_store = AzureSearch(
        azure_search_endpoint=os.environ.get('AZURE_COGNITIVE_SEARCH_ADDRESS'),
        azure_search_key=os.environ.get('AZURE_COGNITIVE_SEARCH_API_KEY'),
        index_name=os.environ.get('AZURE_COGNITIVE_SEARCH_INDEX_NAME'),
        embedding_function=instructor_embeddings.embed_query
    ) 

    application.tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')
    application.model = AutoModelForCausalLM.from_pretrained(
        'stabilityai/stablelm-zephyr-3b',
        device_map="auto"
    )


    @application.route('/user', methods = ['GET'])
    @token_required
    def get_all_users(current_user):

        if not current_user.admin:
            return jsonify({'message': 'You must be an admin to perform that function'})
        
        users = User.query.all()

        output = []

        for user in users:
            user_data = {}
            user_data['public_id'] = user.public_id
            user_data['name'] = user.name
            user_data['password'] = user.password
            user_data['admin'] = user.admin
            output.append(user_data) 

        return jsonify({'users': output})
    


    @application.route('/user/<public_id>', methods = ['GET'])
    @token_required
    def get_one_user(current_user, public_id):
        
        if not current_user.admin:
            return jsonify({'message': 'You must be an admin to perform that function'})


        user = User.query.filter_by(public_id=public_id).first()

        if not user:
            return jsonify({'message': 'No user found'})
        
        user_data = {}
        user_data['public_id'] = user.public_id
        user_data['name'] = user.name
        user_data['password'] = user.password
        user_data['admin'] = user.admin
    
        return jsonify({'user': user_data})
    

    @application.route('/user', methods = ['POST'])
    @token_required
    def create_user(current_user):

        if not current_user.admin:
            return jsonify({'message': 'You must be an admin to perform that function'})


        data = request.get_json()
        
        hashed_password = generate_password_hash(data['password'], method='scrypt')

        new_user = User(public_id = str(uuid.uuid4()), name=data['name'], password=hashed_password, admin=False)
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({'message': 'New user created'})
    


    @application.route('/user/<public_id>', methods = ['PUT'])
    @token_required
    def promote_user(current_user, public_id):
        
        if not current_user.admin:
            return jsonify({'message': 'You must be an admin to perform that function'})


        user = User.query.filter_by(public_id=public_id).first()

        if not user:
            return jsonify({'message': 'No user found'})
        
        user.admin = True
        db.session.commit()


        return jsonify({'message': 'The user has been promoted'})
    


    @application.route('/user/<public_id>', methods = ['DELETE'])
    @token_required
    def delete_user(current_user, public_id):
        
        if not current_user.admin:
            return jsonify({'message': 'You must be an admin to perform that function'})

        user = User.query.filter_by(public_id=public_id).first()

        if not user:
            return jsonify({'message': 'No user found'})
        
        db.session.delete(user)
        db.session.commit()

        return jsonify({'message': 'The user has been deleted'})
    


    @application.route('/login')
    def login():
        auth = request.authorization

        if not auth or not auth.username or not auth.password:
            return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})
        
        user = User.query.filter_by(name=auth.username).first()

        if not user:
            return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required !"'})

        if check_password_hash(user.password, auth.password):
            # exp is expiration date
            token = jwt.encode({'public_id': user.public_id, 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=60)}, app.config['SECRET_KEY'], algorithm="HS256") 

            return jsonify({'token': token}) 
    


        return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required ! !"'})

    #######################################################################################################################################



    @application.route("/api/store_embeddings", methods=["Get"])
    @token_required
    def _store_embeddings(current_user):
        
        if not current_user.admin:
            return jsonify({'message': 'You must be an admin to add embeddings to the database'})

        store_embeddings(application.vector_store, "prefix")


        return jsonify({'message': 'Embeddings successfully added to the database'})



    @application.route("/api/query", methods=["GET"])
    @token_required
    def query(current_user):

        if not current_user.admin:
            return jsonify({'message': 'You must be an admin to query in the database'})


        input_query = request.args.get('query')
        docs = get_similar_content(input_query, application.vector_store)
        
        response = {'content': {}, 'metadata': {}}

        for i, doc in enumerate(docs):
            response['content'][f'response {i+1}:'] = doc.page_content
            response['metadata'][f'source {i+1}:'] = doc.metadata

        return jsonify(response)    
    



    @application.route("/api/query_specific", methods=["GET"])
    @token_required
    def query_specific(current_user):

        if not current_user.admin:
            return jsonify({'message': 'You must be an admin to query a specific document from the database'})

        input_query = request.args.get('query')
        source_file = request.args.get('document')
        
        docs = get_similar_content(input_query, application.vector_store)

        response = {'content': {}, 'metadata': {}}

        for i, doc in enumerate(docs):
            if doc.metadata.get('source') == source_file:
                response['content'][f'response {i+1}:'] = doc.page_content
                response['metadata'][f'source {i+1}:'] = doc.metadata


        return jsonify(response)
    

    @application.route("/api/delete_all", methods=["GET"])
    @token_required
    def delete_all(current_user):

        if not current_user.admin:
            return jsonify({'message': 'You must be an admin to delete all data in the database'})


        try:

            ids = get_all_index_ids()
            delete_index_chunks_by_id(ids)

            response = {'result': 'Successfully deleted all entries from the database'}

            return jsonify(response)

        except Exception as e:
            response = {'result': f'Error: {str(e)}'}
            return jsonify(response)
    
    
    @application.route("/api/delete_specific", methods=["GET"])
    def delete_specific(current_user):  

        if not current_user.admin:
            return jsonify({'message': 'You must be an admin to delete a specific document from the database'})

        try:
        
            source_document = request.args.get('document')
            
            ids = get_specific_index_ids(source_document)
            delete_index_chunks_by_id(ids)


            response = {'result': 'Successfully deleted all entries of the document'}

            return jsonify(response)

        except Exception as e:
            response = {'result': f'Error: {str(e)}'}
            return jsonify(response)
        

        
    @application.route("/api/queryllm", methods=["GET"])
    def queryllm():
        input_query = request.args.get('query')
        
        content = get_model_response(input_query, application.vector_store, application.model, application.tokenizer)


        response = {
            'content': content,
        }


        return jsonify(response)

    return application




if __name__ == '__main__':
    app = create_app()
    app.app_context().push()
    db.create_all()
    app.run(host="0.0.0.0", port=80, threaded=True)