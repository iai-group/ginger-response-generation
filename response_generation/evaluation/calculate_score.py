import pandas as pd

if __name__ == "__main__":
    response_types = ["baseline_top_5", "baseline_top_5_cot", "ginger_top_5", "ginger-fluency_top_5", "ginger-fluency_top_10", "ginger-fluency_top_20", "ginger_lsa_bm25", "ginger_lsa_duot5", "ginger_bertopic_bm25"]
    models = ["gpt4", "claude", "gemini"]

    for response_type in response_types:
        scores = {}
        avg_scores = []

        for model in models:
            results = pd.read_csv("results/nugget_assignment/{}_nugget_assignment_301_topics_{}.csv".format(model, response_type), sep=";")
            scores[model] = results["{}_v_strict_score".format(response_type)].tolist()
            
        avg_scores = [sum([scores[model][i] for model in models])/len(models) for i in range(len(scores[models[0]]))]

        print("Score for response: {}: {}".format(response_type, round(sum(avg_scores)/len(avg_scores), 3)))