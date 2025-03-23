# Response Generation

## Implementation of Response Generation

Generating grounded, completeness-aware responses is a multi-stage process that includes: 
1) detecting information nuggets in top relevant passages, 
2) clustering detected nuggets with respect to different facets of the question, 
3) ranking the clusters with respect to their relevance to the query, 
4) summarizing the top-ranked clusters to be included in a final response,  

The implementation of individual components is provided [here](pipeline/components/). Each file contains exemplary usage of the component. Detailed description of each component is provided [here](pipeline/README.md). The prompts used can be found [here](prompts/README.md).

To run the whole response generation pipeline, use the following command:

``python -m response_generation.pipeline.main ``

Use the following command line arguments to select the setup:
-  *--input_variant* - Variant of the input passages (5_relevant, 5_retrieved, 5_irrelevant, or 5_rephrased). Select between: 5_relevant, 5_irrelevant, 5_retrieved, 5_rephrased, and 5_relevant\breadth-vs-depth.
-  *--information_nuggets* - Variant of the information nuggets (detected_nuggets or information_nuggets). Select between: detected_nuggets and information nuggets (ground-truth nuggets from CAsT-snippets dataset).
-  *--response_variant - Variant of the response generation pipeline (e.g. gpt4-bertopic-duot5-gpt4).
-  *--breadth_vs_depth - Add this flag if you want to run response coverage comparison.
-  *--rephrasing_variant - Add this flag if you want to rephrase the response generated with the pipeline with GPT-4.

Generated data is stored in the CSV file in folder corresponding to the input passage variant used. The name of the file has the following format: 
``*[nugget_detection_method]-[clustering_method]-[ranking_method]-[cluster_summarization_method].csv`` 
with response generated using our pipeline in **cluster_based_response** (or **rephrased_cluster_based_response**) and response generated with the baseline stored in **passages_summary_zero_shot** columns respectively. F

## Implementation of the Baseline

We use approach proposed for open-domain QA in retrieval-augmented setting with the off-the-shelf LLM without further training. We use the most recent at the time of writing snapshot of OpenAI's GPT-4 model, which is the model achieving the highest scores in terms of faithfulness on the task of summary generation, as well as the most commonly used LLM architecture in RAG. The length of generated summary is limited to around 100 words and 3 sentences, which is controlled in task model prompt. Response generation is performed with the most recent snapshot of *GPT-4-turbo (gpt-4-turbo-2024-04-09)* accessed via the OpenAI API. The second baseline uses GPT-4 with Chain-of-Thought prompting  and one ICL demonstration created manually based on TREC CAsT dataset. The implementation of this component can be found in [pipeline/main.py](pipeline/main.py). Make sure to generate the OpenAI API key and add it to the [config.py](config.py) file before running this component. 