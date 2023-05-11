from flask import Flask, request, jsonify
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, StorageContext, load_index_from_storage
# from llama_index.node_parser import SimpleNodeParser
from langchain import OpenAI

import os
import config

app = Flask(__name__)

# initialize index
index = None

@app.before_first_request
def startup():
    initialize_index()

def initialize_index():
    global index
    # set OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

    # load training data from directory {root}/data/*
    documents = SimpleDirectoryReader('data').load_data()

    # parse the document into nodes
    # parser = SimpleNodeParser()
    # nodes = parser.get_nodes_from_documents(documents)

    # index construction
    # index = GPTVectorStoreIndex.from_documents(documents)
    # index = GPTVectorStoreIndex(nodes)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

    # define prompt helper
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )

    # save the index for future use - save ./storage as default
    index.storage_context.persist()

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="storage")

    # load index
    index = load_index_from_storage(
        storage_context, service_context=service_context
    )

@app.route('/')
def home():
    return 'Server is running.'

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        # get parameter
        data = request.get_json()
        query_result = data.get('queryResult')
        query = query_result.get('queryText')

        if query_result.get('action') == 'input.unknown':
            result = query_gpt(query)

        if query_result.get('action') == 'welcome':
            result = "Hi, I am a custom service assistant."
        
        response = {
            'fulfillmentText': result
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify(e), 400
    
def query_gpt(query):
    global index
    # high level API - query the index
    query_engine = index.as_query_engine()

    return query_engine.query(query).response.strip('\n')

if __name__ == '__main__':
    app.run(debug=True)
