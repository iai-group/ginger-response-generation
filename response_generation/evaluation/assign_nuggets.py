import argparse
import ast

import pandas as pd

from response_generation.evaluation.auto_nuggetizer import CLAUDE, GEMINI, GPT


def main(model_name, response_type):
    if model_name == "gpt4":
        llm = GPT()
        llm_determine_nugget_importance = llm
        llm_determine_assign_nuggets = llm
    elif model_name == "claude":
        llm = CLAUDE()
        llm_determine_nugget_importance = llm
        llm_determine_assign_nuggets = llm
    elif model_name == "gemini":
        system_instruction_determine_nugget_importance = "You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query."
        system_instruction_assign_nuggets = "You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets based on if they are captured by a given passage."
        llm_determine_nugget_importance = GEMINI(system_instruction=system_instruction_determine_nugget_importance)
        llm_determine_assign_nuggets = GEMINI(system_instruction=system_instruction_assign_nuggets)

    if response_type in ["baseline_top_5", "ginger_top_5", "ginger-fluency_top_5", "ginger-fluency_top_10", "ginger-fluency_top_20"]:
        responses = pd.read_csv("data/generated_responses/runs_responses.csv", sep=";")
    elif response_type in ["ginger_lsa_bm25", "ginger_lsa_duot5", "ginger_bertopic_bm25"]:
        responses = pd.read_csv("data/generated_responses/ginger_variants_responses.csv", sep=";")
    elif response_type in ["baseline_top_5_cot"]:
        responses = pd.read_csv("data/generated_responses/baseline_cot.csv", sep=";")
    else:
        raise ValueError("Response type not recognized")

    nuggets_automatically_detected = pd.read_csv("results/nugget_detection/" + model_name + "-nugget_detection_301_topics.csv", sep=";")
    
    df_v_strict_scores_per_response_type = {}

    df_nuggets = [[]] * len(responses)
    df_nuggets_importance = [[]] * len(responses)
    df_nuggets_assignments = [[]] * len(responses)
    df_v_strict_score = [-1] * len(responses)

    for n, row in responses.iterrows():
        print("****** {} ******".format(row["qid"]))
        query = row["query"]
        nuggets = ast.literal_eval(nuggets_automatically_detected[nuggets_automatically_detected["qid"] == row["qid"]]["automatically_detected_nuggets"].tolist()[0])
        print(len(nuggets))
        df_nuggets[n] = nuggets

        nuggets_with_importance = llm_determine_nugget_importance.determine_nuggets_importance(query, nuggets)
        nuggets_with_importance = nuggets_with_importance.replace("```python\n", "").replace("\n```", "")
        nuggets_with_importance = ast.literal_eval(nuggets_with_importance)
        print(len(nuggets_with_importance))
        df_nuggets_importance[n] = nuggets_with_importance

        nuggets_assignments = []

        for i in range(0, len(nuggets), 10):
            iteration_nuggets = nuggets[i:i+10]
            nuggets_assigns = llm_determine_assign_nuggets.assign_nuggets(query, row[response_type], iteration_nuggets)
            nuggets_assigns = nuggets_assigns.replace("```python\n", "").replace("\n```", "")
            nuggets_assigns = ast.literal_eval(nuggets_assigns)
            nuggets_assignments.extend(nuggets_assigns)

        print(len(nuggets_assignments))
        df_nuggets_assignments[n] = nuggets_assignments

        assignments_to_consider = []
        for importance, assignment in zip(nuggets_with_importance, nuggets_assignments):
            if importance == "vital":
                assignments_to_consider.append(assignment)
        v_strict_score = 0
        if len(assignments_to_consider) > 0:
            v_strict_score = assignments_to_consider.count("support")/len(assignments_to_consider)
        print(v_strict_score)
        df_v_strict_score[n] = v_strict_score

        responses[response_type + "_automatically_detected_nuggets"] = df_nuggets
        responses[response_type + "_nuggets_importance"] = df_nuggets_importance
        responses[response_type + "_nuggets_assignments"] = df_nuggets_assignments
        responses[response_type + "_v_strict_score"] = df_v_strict_score

        responses.to_csv("results/nugget_assignment/{}_nugget_assignment_{}_topics_{}.csv".format(model_name, len(responses), response_type), sep=";", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assign nuggets to responses')
    parser.add_argument('--model_name', type=str, help='Model name to use for nugget assignment. Chose from: gpt4, claude, gemini')
    parser.add_argument('--response_type', type=str, help='Response type to evaluate. Chose from: baseline_top_5, baseline_top_5_cot, ginger_top_5, ginger-fluency_top_5, ginger-fluency_top_10, ginger-fluency_top_20, ginger_lsa_bm25, ginger_lsa_duot5, ginger_bertopic_bm25')
    args = parser.parse_args()

    main(args.model_name, args.response_type)