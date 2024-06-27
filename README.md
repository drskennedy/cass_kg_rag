# Knowledge Graph with Cassandra Vectorstore in a RAG Setup using LlamaIndex and Mistral 7B for Local Documents

**Step-by-step guide on Medium**: [Evaluating the Impact of Knowledge Graph in RAG Powered By Cassandra Database](https://medium.com/@heelara/evaluating-the-impact-of-knowledge-graph-in-rag-powered-by-cassandra-database-5f7442b4b355)
___
## Context
Retrieval-Augmented Generation (RAG) is a popular technique used to improve the text generation capability of an LLM by keeping it fact driven, but LLM hallucinations continue to be a challenge. Knowledge Graphs of your documents seem to a promising way forth here.
In this project, we will develop a RAG application using `LlamaIndex` pipeline to use vector search powered by Apache Cassandra NoSql database along with an auto-generated knowledge graph using `KnowledgeGraphIndex` to serve as context for a locally hosted Mistral 7B LLM using `llama-cpp-python`.
<br><br>
![System Design](/assets/architecture.png)
___
## How to Install Cassandra Database
- Download and extract the tarball:
```
$ curl -O https://dlcdn.apache.org/cassandra/5.0-beta1/apache-cassandra-5.0-beta1-bin.tar.gz
$ tar xvzf apache-cassandra-5.0-beta1-bin.tar.gz
```
- Move to a desired location, such as /opt/cassandra:
```
$ mv apache-cassandra-5.0-beta1/* /opt/cassandra
```
- Add /opt/cassandra/bin to your shell PATH and source it
- Launch Cassandra as a daemon:
```
$ cassandra -f
```
- Launch Cassandra CLI from a different terminal and create keyspace "vectorstore":
```
$ cqlsh
Connected to Test Cluster at 127.0.0.1:9042
[cqlsh 6.2.0 | Cassandra 5.0-beta1 | CQL spec 3.4.7 | Native protocol v5]
Use HELP for help.

cqlsh> CREATE KEYSPACE vectorstore
  WITH REPLICATION = {
   'class' : 'SimpleStrategy',
   'replication_factor' : 1
  };
cqlsh> quit
```

___
## How to Setup Python virtual environment
- Create and activate the environment:
```
$ python3.11 -m venv kg_qa
$ source kg_qa/bin/activate
```
- Install libraries:
```
$ pip install -r requirements.txt
```
- Download Mistral-7B-Instruct-v0.3.Q2_K.gguf from [MaziyarPanahi HF repo](https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF) to directory `models`.
- Run script `main.py` to start the testing:
```
$ python main.py
```
___
## Quickstart
- To start the app, launch terminal from the project directory and run the following command:
```
$ source kg_qa/bin/activate
$ python main.py
```
- Here is a sample run:
```
$ python main.py
### Cassandra Vectorstore Query ###
Response:  Simplified routing should be enabled in the following scenarios:
- Multiple LAN Subnets which need to be optimized through the Steelhead.
- Multiple VLANS on the LAN segment which need to be optimized through the Steelhead.
Time: 27.895668291999755
================================================================================
### KG Query ###
Response: 1. Simplified routing can be used on SteelHeads when there are no complex routing policies or requirements in place. This is a basic mode that does not require any configuration of routing policies. It simply forwards traffic based on the destination IP address.
2. Simplified routing can also be used when there is a need to bypass complex routing policies, such as when troubleshooting or testing new configurations.
3. However, it's important to note that simplified routing may not provide the same level of performance and efficiency as more advanced routing modes like Optimized Routing or Intelligent Routing. Therefore, it should be used only when simpler routing is sufficient for the network requirements.
Time: 22.151364625000042
================================================================================
### Custom Query ###:
Response:  Simplified routing should be enabled in the following scenarios:
- Multiple LAN Subnets which need to be optimized through the Steelhead.
- Multiple VLANS on the LAN segment which need to be optimized through the Steelhead.
Time: 33.70136175000016
================================================================================
```
___
## Key Libraries
- **LlamaIndex**: Framework for developing applications powered by LLM
- **llama-cpp-python**: Library to load GGUF-formatted LLM from a local directory

___
## Files and Content
- `models`: Directory hosting the downloaded LLM in GGUF format
- `pdf`: Directory hosting the sample niche domain documents
- `main.py`: Main Python script to launch the application
- `custom_retriever.py`: CustomRetriever class incorporating vector and knowledge graph retrievers
- `requirements.txt`: List of Python dependencies (and version)
___

## References
- https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers/
