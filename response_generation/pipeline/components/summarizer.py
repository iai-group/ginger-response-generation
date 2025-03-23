"""Text summarizer."""

from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from openai import OpenAI
from transformers import pipeline

from response_generation.config import DEFAULT_GPT_VERSION, OPENAI_API_KEY
from response_generation.utilities.generation import num_tokens_from_messages
from response_generation.utilities.ranking import Ranking

# _DEFAULT_SUMMARIZER_MODEL = "facebook/bart-large-cnn"
_DEFAULT_SUMMARIZER_MODEL = "Falconsai/text_summarization"


class Summarizer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def summarize_text(self, text: str, max_length: int) -> str:
        """Summarize text.

        Args:
            text: Text to summarize.
            max_length: Maximum number of tokens in the summary.

        Returns:
            Summary of text.
        """
        raise NotImplementedError


class HuggingFaceSummarizer(Summarizer):
    def __init__(self, model_name: str = _DEFAULT_SUMMARIZER_MODEL) -> None:
        """Instantiates a summarizer based on a Hugging Face model.

        The summarization is based on the Pipelines API developed by Hugging
        Face. For more details see:
        https://huggingface.co/docs/transformers/v4.20.0/en/main_classes/pipelines#transformers.SummarizationPipeline #noqa

        Args:
            model_name (optional): Hugging Face model name. Defaults to
              file-level constant _DEFAULT_SUMMARIZER_MODEL.
        """  # noqa
        self._summarizer = pipeline("summarization", model=model_name)

    def summarize_ranking(
        self,
        passages: Ranking,
        k: int = 3,
        min_length: int = 10,
        max_length: int = 250,
    ) -> str:
        """Summarizes passages using a Hugging Face model.

        Args:
            passages: Passages to summarize.
            k (optional): Maximum number of passages to consider for the
              summary.
            min_length (optional): Minimum number of tokens in the summary.
              Defaults to 10 tokens.
            max_length (optional): Maximum number of tokens in the summary.
              Defaults to 250 tokens.

        Returns:
            Abstractive summary of passages.
        """
        topk = passages.fetch_topk_docs(k=k, unique=True)
        texts = list(map(lambda p: p.content, topk))
        text = " ".join(texts)
        summary = self._summarizer(
            text, min_length=min_length, max_length=max_length
        )
        return summary[0]["summary_text"]

    def summarize_text(
        self, text: str, min_length: int = 10, max_length: int = 250,
    ) -> str:
        """Summarizes passages using a Hugging Face model.

        Args:
            text: Text to summarize.
            min_length (optional): Minimum number of tokens in the summary.
              Defaults to 10 tokens.
            max_length (optional): Maximum number of tokens in the summary.
              Defaults to 250 tokens.

        Returns:
            Abstractive summary of passages.
        """
        summary = self._summarizer(
            text, min_length=min_length, max_length=max_length
        )
        return summary[0]["summary_text"]


class GPTSummarizer(Summarizer):
    def __init__(
        self, api_key: str, gpt_version: str = DEFAULT_GPT_VERSION
    ) -> None:
        """Instantiates a summarizer based on OpenAI's GPT model.

        Args:
            api_key: OpenAI API key.
            gpt_version (optional): OpenAI GPT model version. Defaults to
              file-level constant DEFAULT_GPT_VERSION.
        """  # noqa
        self._openai_client = OpenAI(api_key=api_key)
        self._gpt_version = gpt_version

    def summarize_passages(
        self, query: str, passages: str, prompt: str, max_length: int = 300,
    ) -> str:
        """Summarizes passages to answer given query with OpenAI GPT model.

        Args:
            query: Query to answer.
            passages: Passages to summarize.
            prompt: Prompt to use for the OpenAI GPT model.
            max_length (optional): Maximum number of tokens in the summary.
              Defaults to 500 tokens.
            
        Returns:
            Abstractive summary of text as an answer to a query.
        """
        input_sample = {
            "role": "user",
            "content": "Question: {} Passage: {}".format(query, passages),
        }
        if (
            num_tokens_from_messages(
                prompt + [input_sample], model=self._gpt_version
            )
            > 4095
        ):
            return "-1"
        else:
            response = self._openai_client.chat.completions.create(
                model=self._gpt_version,
                messages=prompt + [input_sample],
                max_tokens=max_length,
                seed=13,
            )
            predicted_response = response.choices[0].message.content

        return predicted_response

    def summarize_text(
        self, text: str, prompt: str, max_length: int = 100,
    ) -> str:
        """Summarizes text with OpenAI GPT model.

        Args:
            text: Text to summarize.
            prompt: Prompt to use for the OpenAI GPT model.
            max_length (optional): Maximum number of tokens in the summary.
              Defaults to 250 tokens.
            
        Returns:
            Abstractive summary of text.
        """
        input_sample = {
            "role": "user",
            "content": text,
        }
        if (
            num_tokens_from_messages(
                prompt + [input_sample], model=self._gpt_version
            )
            > 4095
        ):
            return "-1"
        else:
            response = self._openai_client.chat.completions.create(
                model=self._gpt_version,
                messages=prompt + [input_sample],
                max_tokens=max_length,
            )
            predicted_response = response.choices[0].message.content

        return predicted_response


def rephrase_response(
    data: pd.DataFrame, response_type: str, summarizer: Summarizer
) -> List[str]:
    rephrased_responses = []
    prompt = [
        {
            "role": "system",
            "content": "Rephrase the response given a query. Do not change the information included in the response. Do not add information not mentioned in the response.",
        },
    ]
    summarizer = GPTSummarizer(api_key=OPENAI_API_KEY)

    for _, row in data.iterrows():
        rephrased_responses.append(
            summarizer.summarize_text(
                text="Query: "
                + row["query"]
                + " Response: "
                + row[response_type],
                prompt=prompt,
                max_length=300,
            )
        )

    return rephrased_responses


if __name__ == "__main__":
    # Example of response rephrasing with GPT-4
    path = "data/generated_responses/5_relevant/cast-bertopic-bm25-gpt4.csv"
    data = pd.read_csv(path)

    summarizer = GPTSummarizer(api_key=OPENAI_API_KEY)
    rephrased_responses = rephrase_response(
        data, "clusters_based_response", summarizer
    )
    data["rephrased_clusters_based_response"] = rephrased_responses

    data.to_csv(path.replace(".csv", "") + "_rephrased_response.csv")
