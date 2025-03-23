# Response Generation Pipeline

Our proposed response generation pipeline focuses on the inclusion of a maximum number of unique pieces of information answering the question in the generated response, given constraints on the response length. Generating grounded, completeness-aware responses is a multistage process that includes: 
1. detecting information nuggets in top relevant passages, 
2. clustering detected nuggets with respect to different facets of the question, 
3. ranking the clusters with respect to their relevance to the query, 
4. summarizing the top-ranked clusters to be included in a final response, and 

Steps 1-3 aim at curating the context for response generation to mitigate the ``Lost in the Middle'' problem related to LLMs focusing mostly on the beginning and the end of long texts. By operating on information nuggets in all intermediate components of the pipeline we ensure the grounding of the final response in the source passages. In principle, our method ensures that all information in the final response is entailed by the source. 

When developing and further evaluating the proposed method we navigate in the space of conversational information-seeking data for its complexity and numerous challenges. We assume that the de-contextualized query and the ranking of passages with relevance scores are provided as query rewriting and passage retrieval are not the focus of this work. 

To generate responses, run the following command:

```python -m response_generation.pipeline.main --top_n {} --res_length_limit {} --fluency_enhancement {} --baseline --cot --clusterer {} --ranker {}```

Arguments:
- top_n - Number of passages to consider for response generation
- res_length_limit - Maximum length of the response in sentences. Use 100 if you want to limit the response to 400 words.
- fluency_enhancement - Whether to rephrase the generated responses
- baseline - Add this flag to generate baseline responses
- cot - Add this flag to generate baseline responses using the CoT model
- clusterer - Clustering model to use for response generation. Choose from: bertopic, lsa
- ranker - Ranking model to use for response generation. Choose from: duot5, bm25

To detect nuggets in the passages, run the following command:

```python -m response_generation.pipeline.detect_nuggets```


## Information Nugget Detection

We aim to automatically detect information nuggets using an LLM. The LLM is prompted to annotate information nuggets containing the key information that answers the query within a given passage. It is instructed to copy the text of the passage and place the annotated information nuggets between specific tags, without modifying the passage content or adding any extra symbols, spaces, etc.

Automatic detection of information nuggets is performed with GPT-4 provided with a query and a passage. The implementation of this component can be found in [components/nugget_detection.py](components/nugget_detection.py). Make sure to generate the OpenAI API key and add it to the [config.py](../config.py) file before running this component. 


## Clustering Information Nuggets

After detecting all information nuggets that contain at least one piece of information answering the query we proceed to clustering them with respect to different facets of the topic that they cover. 

Experiments has been run with HugginFace SentenceTransformers with [DBSCAN clustering](https://github.com/huggingface/text-clustering), k-Means, and [aglomerative clustering](https://www.sbert.net/examples/applications/clustering/README.html), as well as topic modelling with LDA, SpaCy semantic similarity and [DBSCAN clustering algorithm](https://towardsdatascience.com/nlp-with-lda-latent-dirichlet-allocation-and-text-clustering-to-improve-classification-97688c23d98).

The best results are achieved with [BERTopic](https://www.pinecone.io/learn/bertopic/) with parameters for [UMAP](https://umap-learn.readthedocs.io/en/latest/parameters.html) and [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/parameter\_selection.html) set experimentally on a number of samples selected randomly from validation partition. Additionally, we take into account another variant of clustering component based on latent semantic analysis [LSA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) that represents more classic approach compared to BERTopic. The implementation of this component can be found in [components/clustering.py](components/clustering.py).


## Ranking Information Clusters

Information nuggets in clusters are joined and treated as individual passages that are ranked in terms of their relevance to the query. We implement pairwise reranking based on duoT5 and a more classic approach based on BM25 for comparison. Our implementation of duoT5 is based on the HuggingFace transformers library and the *castorini/duot5-base-msmarco* model published on [HuggingFace](https://huggingface.co/castorini/duot5-base-msmarco). BM25 implementation, representing more classical approach in the experiments, utilizes [rank-bm25](https://pypi.org/project/rank-bm25/) package. The implementation of this component can be found in [components/ranker.py](components/ranker.py).


## Summarizing Information Nuggets

The final response is the result of merging the summaries of the top *n* clusters. Following post-retrieval processes implemented in [HayStack (LostInTheMiddleRanker)](https://towardsdatascience.com/enhancing-rag-pipelines-in-haystack-45f14e2bc9f5), we make sure that the most relevant information is the one the generation process is focused on. The information nuggets clusters that are ranked as the highest (top 3) are passed to query biased summarization~performed with the most recent snapshot of *GPT-4-turbo (gpt-4-turbo-2024-04-09)* accessed via the OpenAI API. The length of each cluster summary is limited to one sentence and around 35 words and specified in the prompt. The output length limit is specified in the models parameters. Information clustering should yield at least two clusters, each containing two information nuggets. If the number of snippets identified in the top n passages is less than four, we cannot directly apply our proposed method. To maintain comparability in response length between our method and the baseline, we skip the clustering step for queries with two or three identified information nuggets and treat each information nugget as an independent cluster. Cluster ranking and summarization are then performed in the standard way. Queries with only one information nugget that result in a one-sentence answer are not comparable to the baseline response and are excluded from the evaluation. Similarly, queries without any identified information nuggets are excluded from the method evaluation, as they fall under the problem of query unanswerability, which is beyond the scope of this paper. The implementation of this component can be found in [components/summarizer.py](components/summarizer.py).  Make sure to generate the OpenAI API key and add it to the [config.py](../config.py) file before running this component.