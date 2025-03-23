"""Class for detecting information nuggets in passage given a query."""

import ast
import re
from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from openai import OpenAI

from response_generation.config import DEFAULT_GPT_VERSION, OPENAI_API_KEY
from response_generation.utilities.generation import num_tokens_from_messages

_DEFAULT_PROMPT = [
    {
        "role": "system",
        "content": (
            "Given a query and a passage, annotate information nuggets "
            "that contain the key information answering the query. Copy the "
            "text of the passage and put the annotated information nuggets "
            "between <IN> and </IN>. Do NOT modify the content of the passage. "
            "Do NOT add additional symbols, spaces, etc. to the text."
        ),
    },
]


class NuggetDetector(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def detect_nuggets(self, query: str, passage: str,) -> List[str]:
        """Detects information nuggets in a passage given a query.

        Args:
            query: Query to answer.
            passage: Passage to detect information nuggets in.

        Returns:
            List of information nuggets extracted from the passage.
        """
        raise NotImplementedError


class GPTNuggetDetector(NuggetDetector):
    def __init__(
        self, api_key: str, gpt_version: str = DEFAULT_GPT_VERSION
    ) -> None:
        """Instantiates a nugget detector using OpenAI GPT model.

        Args:
            api_key: OpenAI API key.
            gpt_version (optional): OpenAI GPT model version. Defaults to
              file-level constant DEFAULT_GPT_VERSION.
        """  # noqa
        self._openai_client = OpenAI(api_key=api_key)
        self._gpt_version = gpt_version

    def detect_nuggets(
        self, query: str, passage: str, prompt: str = _DEFAULT_PROMPT,
    ) -> List[str]:
        """Detects information nuggets in a passage given a query.

        Args:
            query: Query to answer.
            passage: Passage to detect information nuggets in.
            prompt (optional): Prompt to use for the OpenAI GPT model.
              Defaults to file-level constant _DEFAULT_PROMPT.

        Returns:
            List of information nuggets extracted from the passage.
        """
        input_sample = {
            "role": "user",
            "content": "Question: {} Passage: {}".format(query, passage),
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
                seed=13,
            )
            annotated_passage = response.choices[0].message.content
            return re.findall("<IN>(.*?)</IN>", annotated_passage)


if __name__ == "__main__":
    # Example usage
    nugget_detector = GPTNuggetDetector(api_key=OPENAI_API_KEY)

    query = "What happens when all corals die?"
    passage = (
        "When bleaching occurs, the reefs lose much of their "
        "characteristic color as the algae and the coral animals die if loss of "
        "the symbiotic zooxanthellae is prolonged. Rising levels of atmospheric "
        "carbon dioxide further threaten the corals in other ways; as CO 2 "
        "dissolves in ocean waters, it lowers the pH and increases ocean "
        "acidity.  As acidity increases, it interferes with the calcification "
        "that normally occurs as coral animals build their calcium carbonate "
        "homes.  When a coral reef begins to die, species diversity plummets as "
        "animals lose food and shelter.  Coral reefs are also economically "
        "important tourist destinations, so the decline of coral reefs poses a "
        "serious threat to coastal economies.  Human population growth has "
        "damaged corals in other ways, too.  As human coastal populations "
        "increase, the runoff of sediment and agricultural chemicals has "
        "increased, too, causing some of the once-clear tropical waters to "
        "become cloudy.  At the same time, overfishing of popular fish species "
        "has allowed the predator species that eat corals to go unchecked. "
        "Although a rise in global temperatures of 1–2˚C (a conservative "
        "scientific projection) in the coming decades may not seem large, it is "
        "very significant to this biome.  When change occurs rapidly, species "
        "can become extinct before evolution leads to new adaptations."
    )

    information_nuggets = nugget_detector.detect_nuggets(query, passage)

    for information_nugget in information_nuggets:
        print(information_nugget)

    # Automatic nugget detection for test partition

    data_sample = pd.read_csv(
        "data/input_passages/5_relevant-cast-snippets.csv"
    )
    detected_nuggets = []

    nugget_detector = GPTNuggetDetector(api_key=OPENAI_API_KEY)

    for current_query_id in list(data_sample["query_id"]):
        information_nuggets_df = information_nuggets[
            information_nuggets["query_id"] == current_query_id
        ]
        query = information_nuggets_df["query"].values[0]
        query_id = information_nuggets_df["query_id"].values[0]
        passages = information_nuggets_df["passage"].values[0]
        passages_no_annotations = [
            passage.replace("<IN>", "").replace("</IN>", "")
            for passage in passages
        ]
        detected_information_nuggets = []
        for passage in passages_no_annotations:
            detected_information_nuggets.extend(
                nugget_detector.detect_nuggets(query, passage)
            )
        detected_nuggets.append(detected_information_nuggets)

    data_sample["detected_nuggets"] = detected_nuggets
    data_sample.to_csv(
        "data/input_passages/5_relevant-_detected_nuggets.csv", index=False
    )
