from llama_index.core import (
    VectorStoreIndex,
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings,
    get_response_synthesizer,
)
from llama_index.vector_stores.cassandra import CassandraVectorStore
from cassandra.cluster import Cluster
import cassio
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.retrievers import (
   VectorIndexRetriever,
    KGTableRetriever,
)
from custom_retriever import CustomRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import timeit
import datetime

# connect to cassandra
cluster = Cluster(["127.0.0.1"])
session = cluster.connect("vectorstore")
cassio.init(session=session, keyspace="vectorstore")

llm = LlamaCPP(
    model_path='./models/mistral-7b-instruct-v0.3.Q2_K.gguf',
    temperature=0.1,
    max_new_tokens=256,
    context_window=2048,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    model_kwargs={"n_gpu_layers": 1},
    verbose=False
)
embed_model = HuggingFaceEmbedding()
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

cassandra_store = CassandraVectorStore(
    table="cass_v_table", embedding_dimension=384
)

# try reading from the Cassandra DB
vector_index = VectorStoreIndex.from_vector_store(vector_store=cassandra_store)
# run a test query
if vector_index.as_query_engine().query("test query").response == "Empty Response":
    storage_context = StorageContext.from_defaults(vector_store=cassandra_store)
    documents = SimpleDirectoryReader("./pdf/").load_data()
    vector_index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context,embed_model=embed_model
    )

try:
    # check if index on disk
    gstorage_context = StorageContext.from_defaults(persist_dir='./storage')
    kg_index = load_index_from_storage(storage_context = gstorage_context)
except Exception:
    graph_store = SimpleGraphStore()
    gstorage_context = StorageContext.from_defaults(graph_store=graph_store)
    documents = SimpleDirectoryReader("./pdf/").load_data()
    start = timeit.default_timer()
    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=gstorage_context,
        max_triplets_per_chunk=10,
        include_embeddings=True,
    )
    kg_gen_time = timeit.default_timer() - start # seconds
    print(f'KG generation completed in: {datetime.timedelta(seconds=kg_gen_time)}')
    # save to disk
    gstorage_context.persist()

'''
## create graph
from pyvis.network import Network
g = kg_index.get_networkx_graph(200)
net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(g)
net.show("steelhead_kb.html")
'''

# create retrievers
vector_retriever = VectorIndexRetriever(index=vector_index)
kg_retriever = KGTableRetriever(index=kg_index, retriever_mode='keyword', include_text=False)
custom_retriever = CustomRetriever(vector_retriever, kg_retriever)

# create response synthesizer
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

# create query engines
vector_query_engine = vector_index.as_query_engine()

kg_query_engine = kg_index.as_query_engine(
    # setting to false uses the raw triplets instead of adding the text from the corresponding nodes
    include_text=False,
    retriever_mode='keyword',
    response_mode="tree_summarize",
)

custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)

query = "When should simplified routing be used on SteelHeads?"

# cassandra vectorstore
start = timeit.default_timer()
response = vector_query_engine.query(query)
vs_qa_resp_time = timeit.default_timer() - start # seconds
print(f'### Cassandra Vectorstore Query ###\nResponse: {response.response}\nTime: {vs_qa_resp_time:.3f}\n{"="*80}')

# KG query
start = timeit.default_timer()
response = kg_query_engine.query(query)
kg_qa_resp_time = timeit.default_timer() - start # seconds
print(f'### KG Query ###\nResponse: {response.response}\nTime: {kg_qa_resp_time:.3f}\n{"="*80}')

# custom query
start = timeit.default_timer()
response = custom_query_engine.query(query)
c_qa_resp_time = timeit.default_timer() - start # seconds
print(f'### Custom Query ###:\nResponse: {response.response}\nTime: {c_qa_resp_time:.3f}\n{"="*80}')

