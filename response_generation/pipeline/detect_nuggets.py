import ast
import json

import pandas as pd
from transformers import set_seed

from response_generation.config import OPENAI_API_KEY
from response_generation.pipeline.components.nugget_detection import GPTNuggetDetector
from response_generation.utilities.ranking import Query, Ranking, ScoredDocument

set_seed(42)


if __name__ == "__main__":
    top_n = 20

    test_data = []
    with open("data/input_passages/retrieve_results_fs4_bm25+rocchio_snowael_snowaem_gtel+monot5_rrf+rz_rrf.rag24.test_top100.jsonl") as file:
        for json_obj in file:
            test_query = json.loads(json_obj)
            test_data.append(test_query)

    data_sample = pd.DataFrame()
    data_sample["query_id"] = [sample["query"]["qid"] for sample in test_data]
    data_sample["query"] = [sample["query"]["text"] for sample in test_data]
    data_sample["passage_id"] = [[candidate["docid"] for candidate in sample["candidates"][:top_n]] for sample in test_data]
    data_sample["passage"] = [[candidate["doc"]["segment"] for candidate in sample["candidates"][:top_n]] for sample in test_data]

    information_nuggets = []
    nugget_detector = GPTNuggetDetector(api_key=OPENAI_API_KEY)
    
    for _, row in data_sample.iterrows():
        query = row["query"]
        query_id = row["query_id"]
        passages = row["passage"]
        print("ID: " + query_id + ": " + query)

        information_nuggets_per_doc = {}
        for passage, passage_id in zip(passages, row["passage_id"]):
            detected_nuggets = nugget_detector.detect_nuggets(query, passage)
            information_nuggets_per_doc[passage_id] = detected_nuggets
        
        information_nuggets.append(information_nuggets_per_doc)
        data_sample["information_nuggets"] = information_nuggets + [{}] * (len(data_sample) - len(information_nuggets))
        data_sample.to_csv("data/input_passages/top_" + str(top_n) + "-nuggets.csv",index=False)

    data_sample["information_nuggets"] = information_nuggets
    data_sample.to_csv("data/input_passages/top_" + str(top_n) + "-nuggets.csv",index=False)