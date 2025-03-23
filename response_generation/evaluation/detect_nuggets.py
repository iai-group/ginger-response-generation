"""Script to detect nuggets in documents using a language model."""
import argparse

import pandas as pd

from response_generation.evaluation.auto_nuggetizer import CLAUDE, GEMINI, GPT


def main(model_name):
    
    if model_name == "gpt4":
        llm = GPT()
    elif model_name == "claude":
        llm = CLAUDE()
    elif model_name == "gemini":
        system_instruction_detect_nuggets = "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query."
        llm = GEMINI(system_instruction=system_instruction_detect_nuggets)

    relevant_documents = pd.read_csv("data/input_passages/qrels.rag24.test-umbrela-all_doc_content.txt", sep="\t")
    segments_number = 10

    queries = pd.read_csv("data/rag_test.csv", sep="\t", header=None)
    queries.columns = ["qid", "query"]
    print(len(queries))

    df_nuggets = ["[]"] * len(queries)

    for n, row in queries.iterrows():
        print("****** {} ******".format(n))
        query = row["query"]
        docs = relevant_documents[relevant_documents["qid"] == row["qid"]]["docs"].tolist()
        print(row["qid"], len(docs))

        if df_nuggets[n] != "[]":
            continue

        nuggets = []

        for i in range(0, len(docs), segments_number):
            iteration_docs = docs[i:i+segments_number]
            new_nuggets = llm.detect_nuggets(query, iteration_docs, nuggets)
            new_nuggets = new_nuggets.replace("```python\n", "").replace("\n```", "")
            nuggets = new_nuggets

        df_nuggets[n] = nuggets
    
        queries["automatically_detected_nuggets"] = df_nuggets
        queries.to_csv("results/nugget_detection/" + model_name + "-nugget_detection_" + str(len(df_nuggets)) + "_topics.csv", sep=";", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect nuggets in documents')
    parser.add_argument('--model_name', type=str, help='Model name to use for nugget detection. Chose from: gpt4, claude, gemini')
    args = parser.parse_args()
    
    main(args.model_name)