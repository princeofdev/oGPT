from flask import Flask, request, jsonify
import os
import sys
import config
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, StorageContext, load_index_from_storage
from llama_index.optimization.optimizer import SentenceEmbeddingOptimizer
from langchain import OpenAI
from flask_caching import Cache

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple', 'CACHE_DEFAULT_TIMEOUT': 60*60})

# Load training data
def load_document():
    return SimpleDirectoryReader('data').load_data()

# Initialize the index and storage context
service_context = None
index = None
storage_context = None

def initialize_index():
    global service_context, index, storage_context

    os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    documents = load_document()

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist()

    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context, service_context=service_context)

# call initialize_index function when app starts up
@app.before_first_request
async def startup():
    initialize_index()

@app.route('/')
def home():
    return 'Server is running.'

@app.route('/webhook', methods=['POST'])
async def webhook():
    
    try:
        # get parameter
        data = request.get_json()
        query_result = data.get('queryResult')
        query = query_result.get('queryText')
        
        if query_result.get('action') == 'input.unknown':
            # use cache to store query responses
            response = cache.get(query)
            if response is None:
                response = await query_gpt(query)
                cache.set(query, response)

        if query_result.get('action') == 'welcome':
            response = "Hi, I am a custom service assistant."

        result = {
            'fulfillmentText': response
        }
        return jsonify(result)
    
    except Exception as e:
        print('error', e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print('oops', exc_type, fname, exc_tb.tb_lineno)
        return '', 400

async def query_gpt(query):
    global index

    # high-level API - query the index
    query_engine = index.as_query_engine(
        optimizer=SentenceEmbeddingOptimizer(percentile_cutoff=0.5)
    )

    return query_engine.query(query).response.strip('\n')

if __name__ == '__main__':
    app.run(debug=True)
