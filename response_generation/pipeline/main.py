import argparse
import ast

import pandas as pd
from transformers import set_seed

from response_generation.config import OPENAI_API_KEY
from response_generation.pipeline.components.clustering import (
    BERTopicClustering,
    LSAClustering,
)
from response_generation.pipeline.components.nugget_detection import GPTNuggetDetector
from response_generation.pipeline.components.ranker import BM25Reranker, DuoT5Reranker
from response_generation.pipeline.components.summarizer import (
    GPTSummarizer,
    rephrase_response,
)
from response_generation.utilities.ranking import Query, Ranking, ScoredDocument

set_seed(42)

def main(top_n, res_length_limit, fluency_enhancement, baseline, cot, clusterer, ranker):
    data_sample = pd.read_csv("trec-rag/top_20-nuggets.csv")

    queries = []
    queries_ids = []
    passages_used = []
    passages_used_ids = []
    information_nuggets = []
    retrieved_clusters = []
    ranked_clusters = []
    clusters_summaries = []
    clusters_based_responses = []
    passages_summary_zero_shot = []

    ginger_version = "baseline_cot_3"
    nugget_detector = GPTNuggetDetector(api_key=OPENAI_API_KEY)
    if clusterer == "bertopic":
        clusterer = BERTopicClustering()
    elif clusterer == "lsa":    
        clusterer = LSAClustering()
    if ranker == "duot5":
        ranker = DuoT5Reranker()
    elif ranker == "bm25":
        ranker = BM25Reranker()
    summarizer = GPTSummarizer(api_key=OPENAI_API_KEY)

    for _, row in data_sample.iterrows():
        data_so_far = pd.DataFrame()
        query = row["query"]
        queries.append(query)
        query_id = row["query_id"]
        queries_ids.append(query_id)
        passages = ast.literal_eval(row["passage"])[:top_n]
        passages_used.append(passages)
        passage_ids = ast.literal_eval(row["passage_id"])[:top_n]
        passages_used_ids.append(passage_ids)

        information_nuggets_per_doc = {}
        query_information_nuggets = []
        for passage, passage_id in zip(passages, row["passage_id"]):
            detected_nuggets = nugget_detector.detect_nuggets(query, passage)
            information_nuggets_per_doc[passage_id] = detected_nuggets
            query_information_nuggets.extend(detected_nuggets)
        
        information_nuggets.append(information_nuggets_per_doc)

        print("------------------")
        print("Generating response for query: " + query + " ID: " + query_id)
        if len(query_information_nuggets) <= 1:
            retrieved_clusters.append([])
            ranked_clusters.append([])
            clusters_summaries.append([])
            clusters_based_responses.append([])
            if baseline:
                passages_summary_zero_shot.append([])
        else:
            if len(query_information_nuggets) >= 4:
                clusters = clusterer.cluster(query_information_nuggets)
                information_nugget_clusters = []
                for cluster_id in list(set(clusters["Topic"])):
                    cluster_docs = list(
                        clusters[clusters["Topic"] == cluster_id]["Document"]
                    )
                    doc = ScoredDocument(
                        cluster_id, "; ".join(cluster_docs), len(cluster_docs)
                    )
                    information_nugget_clusters.append(doc)
            else:
                information_nugget_clusters = []
                for cluster_id in range(0, len(query_information_nuggets)):
                    doc = ScoredDocument(
                        cluster_id, query_information_nuggets[cluster_id]
                    )
                    information_nugget_clusters.append(doc)

            retrieved_clusters.append(
                [
                    (cluster.doc_id, cluster.content)
                    for cluster in information_nugget_clusters
                ]
            )

            information_nuggets_ranking = Ranking(
                query_id=query_id, scored_docs=information_nugget_clusters
            )
            clusters_ranking = ranker.rerank(
                Query(query_id, query),
                information_nuggets_ranking,
                len(clusters),
            )
            clusters_ranking_docs = clusters_ranking.documents()
            ranked_clusters.append(
                [
                    (cluster_id, cluster_content)
                    for cluster_id, cluster_content in zip(
                        clusters_ranking_docs[0], clusters_ranking_docs[1]
                    )
                ]
            )

            prompt_snippets = [
                {
                    "role": "system",
                    "content": "Summarize the provided information into one sentence (approximately 35 words). Generate one-sentence long summary that is short, concise and only contains the information provided.",
                },
            ]

            summaries = []
            length_limit_reached = False
            for cluster_id, cluster_content in zip(
                clusters_ranking_docs[0], clusters_ranking_docs[1]
            ):
                if not length_limit_reached:
                    cluster_summary = summarizer.summarize_text(
                        text="Provided information: " + cluster_content,
                        prompt=prompt_snippets,
                    )
                    summaries.append(cluster_summary)
        
                    response_till_now = " ".join(summaries)

                    if len(response_till_now.split()) > 400 or len(summaries) > res_length_limit:
                        summaries = summaries[:-1]
                        length_limit_reached = True
        
            clusters_summaries.append(summaries)
            clusters_based_responses.append(" ".join(summaries[:res_length_limit]))
            
            # Baseline response generation prompt
            if baseline:
                if cot:
                    icl_query = "Tell me more about the Blue Lives Matter movement."
                    icl_context = """The internet facilitates the spread of the message 'All Lives Matter' as a response to the Black Lives Matter hashtag as well as the 'Blue Lives Matter' hashtag as a response to Beyonce's halftime performance speaking out against police brutality.\nFollowing the shooting of two police officers in Ferguson and in response to BLM, the hashtag [[Blue Lives Matter|#BlueLivesMatter]] was created by supporters of the police. Following this, Blue Lives Matter became a pro-police movement in the United States. It expanded after the killings of American police officers.\nOn December 20, 2014, in the wake of the killings of officers Rafael Ramos and Wenjian Liu, a group of law enforcement officers formed Blue Lives Matter to counter media reports that they perceived to be anti-police. Blue Lives Matter is made up of active and retired law enforcement officers. The current national spokesman for Blue Lives Matter is retired Las Vegas Metropolitan Police Department Lieutenant Randy Sutton.\nOriginating in New York City in December 2014, Blue Lives Matter NYC is an organization and current nationwide movement that was created to help law enforcement officers and their families during their times of need. Sergeant Joey Imperatrice, along with Officers Chris Brinkley and Carlos Delgado, felt 'compelled to show support for their brothers and sisters in blue' and handed out bracelets that stated 'Blue Lives Matter'. They decided to create an organization, which became the non-profit Blue Lives Matter NYC. This organization's mission is to raise awareness and encourage public aid for the needs of police officers, to help police officers assist one another, and to provide a police officer's family with comfort and support during difficult times. This campaign is designed to raise both awareness and money for families in need. In order to increase nationwide awareness, over three hundred billboards have been posted with the slogan 'Blue Lives Matter'. Many of these billboards are also accompanied by the hashtag #thankublu, which individuals use to show their support for police officers.\nBlue Lives Matter is a pro-police movement in the United States. It was started after the killings of NYPD officers Rafael Ramos and Wenjian Liu in Brooklyn, New York, on December 20, 2014, after they were ambushed in their patrol car. Blue Lives Matter was formed in reaction to the Black Lives Matter movement, which seeks to end police brutality against the African American community."""
                    icl_nuggets = "'Blue Lives Matter' hashtag as a response to Beyonce's halftime performance speaking out against police brutality\n[[Blue Lives Matter|#BlueLivesMatter]] was created by supporters of the police\nBlue Lives Matter became a pro-police movement in the United States\na group of law enforcement officers formed Blue Lives Matter to counter media reports that they perceived to be anti-police\nmade up of active and retired law enforcement officers\nhelp law enforcement officers and their families during their times of need\nraise awareness and encourage public aid for the needs of police officers, to help police officers assist one another, and to provide a police officer's family with comfort and support as they go through hard times\nraise both awareness and money for the families in need\nover three hundred billboards have been posted with the slogan 'Blue Lives Matter'\n#thankublu, a hashtag many individuals use to show their support for police officers\nIt was started after the killings of NYPD officers Rafael Ramos and Wenjian Liu in Brooklyn, New York, on December 20, 2014, after they were ambushed in their patrol car\n which seeks to end police brutality against the African American community"
                    icl_clusters = "Group A – Purpose & Support for Officers:\nhelp law enforcement officers and their families during their times of need\nraise awareness and encourage public aid for the needs of police officers, to help police officers assist one another, and to provide a police officer's family with comfort and support as they go through hard times\nraise both awareness and money for the families in need\nGroup B - Origin & Formation:\n'Blue Lives Matter became a pro-police movement in the United States\na group of law enforcement officers formed Blue Lives Matter to counter media reports that they perceived to be anti-police\nmade up of active and retired law enforcement officers\nIt was started after the killings of NYPD officers Rafael Ramos and Wenjian Liu in Brooklyn, New York, on December 20, 2014, after they were ambushed in their patrol car\nwhich seeks to end police brutality against the African American community\nGroup C– Broader Context & Media Reaction:\n'Blue Lives Matter' hashtag as a response to Beyonce's halftime performance speaking out against police brutality\n[[Blue Lives Matter|#BlueLivesMatter]] was created by supporters of the police\n#thankublu, a hashtag many individuals use to show their support for police officers\nover three hundred billboards have been posted with the slogan 'Blue Lives Matter'"
                    icl_ranking = "Group B – Origin & Formation (Most relevant)\nGroup A – Purpose & Support for Officers\nGroup C – Broader Context & Media Reaction"
                    icl_response = "Blue Lives Matter was founded by active and retired law enforcement officers following the targeted killings of NYPD officers Rafael Ramos and Wenjian Liu on December 20, 2014. It emerged as a pro-police movement aimed at countering what it perceived as anti-police media narratives while supporting officers and their families through fundraising and awareness campaigns. The movement also engages in public outreach through billboards, social media hashtags like #thankublu, and nonprofit initiatives dedicated to aiding law enforcement personnel in times of need."

                    icl_demo = "Question: {}\n\nPassage: [passage]{}[/passage]\n\nStep-by-step response generation process:\n\nStep 1: Extract key pieces of information relevant to the query\n{}\n\nStep 2: Group related pieces of information\n{}\n\nStep 3: Rank groups based on relevance to the query\n{}\n\nStep 4: Generate a coherent, concise response\nResponse: {}".format(icl_query, icl_context, icl_nuggets, icl_clusters, icl_ranking, icl_response)
                    prompt_passages_zero_shot = [
                        {
                            "role": "system",
                            "content": "TASK\n\nYou are an assistant generating responses to user questions based on the provided information. Your response should rely on the context passage and it should not incorporate any additional information.\n\nSPECIFIC STEPS\n\nFollow a structured step-by-step approach to ensure relevance and coherence in the generated response:\n- Step 1: extract key pieces of information relevant to the question from the provided passage\n- Step 2: group related pieces if information based on the aspect of the topic they discuss or the point of view they represent\n- Step 3: rank groups of information based on their relevance to the query\n- Step 4: use the top groups of relevant information to write a coherent and concise response\n\nThe final response should be three sentences long (approximately 100 words).\n\nEXAMPLE\n\n{}\n\nNOW PERFORM THE TASK ON THE FOLLOWING INPUT\n\n".format(icl_demo),
                        },
                    ]
                else:
                    prompt_passages_zero_shot = [
                        {
                            "role": "system",
                            "content": "Generate the answer to a query that is 3 sentences long (approximately 100 words in total) using the provided information. Use only the provided information and do not add any additional information.",
                        },
                    ]

                passages_summary_zero_shot.append(
                    summarizer.summarize_passages(
                        query=query,
                        passages="".join(passages),
                        prompt=prompt_passages_zero_shot,
                        max_length=1000,
                    )
                )
        
        data_so_far["query_id"] = queries_ids
        data_so_far["query"] = queries
        data_so_far["passage_id"] = passages_used_ids
        data_so_far["passage"] = passages_used
        data_so_far["information_nuggets"] = information_nuggets
        data_so_far["clusters"] = retrieved_clusters
        data_so_far["ranked_clusters"] = ranked_clusters
        data_so_far["clusters_summaries"] = clusters_summaries
        data_so_far["clusters_based_response"] = clusters_based_responses
        if baseline:
            if cot:
                data_so_far["baseline_top_5_cot"] = passages_summary_zero_shot
            else:
                data_so_far["baseline_top_5"] = passages_summary_zero_shot
        
        if res_length_limit < 50:
            data_so_far.to_csv("data/generated_responses/" + ginger_version + "_top_+ " + str(top_n) + "-" + str(res_length_limit) + "_sentences_max" + "-output.csv",index=False)
        else:
            data_so_far.to_csv("trec-rag/" + ginger_version + "_top_+ " + str(top_n) + "-400_words_max" + "-output.csv",index=False)

    data_sample["passage_id"] = passages_used_ids
    data_sample["passage"] = passages_used
    data_sample["information_nuggets"] = information_nuggets
    data_sample["clusters"] = retrieved_clusters
    data_sample["ranked_clusters"] = ranked_clusters
    data_sample["clusters_summaries"] = clusters_summaries
    data_sample["clusters_based_response"] = clusters_based_responses
    if fluency_enhancement:
        data_sample["rephrased_clusters_based_response"] = rephrase_response(
            data_sample, "clusters_based_response", summarizer
        )
    if baseline:
        if cot:
            data_so_far["baseline_top_5_cot"] = passages_summary_zero_shot
        else:
            data_so_far["baseline_top_5"] = passages_summary_zero_shot
    
    if res_length_limit < 50:
        data_sample.to_csv("trec-rag/" + ginger_version + "_top_+ " + str(top_n) + "-" + str(res_length_limit) + "_sentences_max" + "-output_all.csv",index=False)
    else:
        data_sample.to_csv("trec-rag/" + ginger_version + "_top_+ " + str(top_n) + "-400_words_max" + "-output_all.csv",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate responses for TREC-RAG queries')
    parser.add_argument('--top_n', type=int, help='Number of passages to consider for response generation')
    parser.add_argument('--res_length_limit', type=int, help='Maximum length of the response in sentences. Use 100 if you want to limit the response to 400 words.')
    parser.add_argument('--fluency_enhancement', action='store_true', help='Whether to rephrase the generated responses')
    parser.add_argument('--baseline', action='store_true', help='Whether to generate baseline responses')
    parser.add_argument('--cot', action='store_true', help='Whether to generate baseline responses using the CoT model')
    parser.add_argument('--clusterer', type=str, help='Clustering model to use for response generation. Choose from: bertopic, lsa')
    parser.add_argument('--ranker', type=str, help='Ranking model to use for response generation. Choose from: duot5, bm25')

    args = parser.parse_args()

    main(args.top_n, args.res_length_limit, args.fluency_enhancement, args.baseline, args.cot, args.clusterer, args.ranker)