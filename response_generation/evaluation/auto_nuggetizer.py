import ast
import json
import time
from abc import ABC
from typing import List

import anthropic
import google
import google.generativeai as genai
import pandas as pd
import requests
from openai import OpenAI

from response_generation.config import CLAUDE_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

class LLM(ABC):
    def __init__(self) -> None:
        pass


class GPT(LLM):
    def __init__(
        self, api_key: str = OPENAI_API_KEY, gpt_version: str = "gpt-4o-2024-08-06"
    ) -> None:
        """Instantiates an LLM based on OpenAI's GPT model.

        Args:
            api_key: OpenAI API key.
            gpt_version (optional): OpenAI GPT model version. Defaults to
              file-level constant DEFAULT_GPT_VERSION.
        """  # noqa
        self._openai_client = OpenAI(api_key=api_key)
        self._gpt_version = gpt_version
        

    def detect_nuggets(
        self, query: str, docs: List[str], nuggets: List[str], max_length: int = 1000,
    ) -> List[str]:
        prompt = [
            {
                "role": "system",
                "content": "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query.",
            },
        ]

        user_instruction = "Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process). Return only the final list of all nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. Ensure the updated nugget list has at most 30 nuggets (can be less), keeping only the most vital ones. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information."
        final_comment = 'Only update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions). List in the form ["a", "b", ...] and a and b are strings with no mention of "'
        context = ""
        for i, doc in enumerate(docs):
            context = context + "[{}] {}\n".format(i+1, doc)

        input_sample = {
            "role": "user",
            "content": "{}\nSearch Query: {}\nContext: {}Search Query: {}\nInitial Nugget List: {}\nInitial Nugget List Length: {}\n{}\nUpdated Nugget List:".format(user_instruction, query, context, query, str(nuggets), len(nuggets), final_comment),
        }
        response = self._openai_client.chat.completions.create(
            model=self._gpt_version,
            messages=prompt + [input_sample],
            max_tokens=max_length,
            temperature=0.0,
        )
        predicted_response = response.choices[0].message.content

        return predicted_response
    
    def determine_nuggets_importance(
        self, query: str, nuggets: List[str], max_length: int = 1000,
    ) -> List[str]:
        prompt = [
            {
                "role": "system",
                "content": "You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query.",
            },
        ]
        user_instruction = "Based on the query, label each of the {} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a “good” answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.".format(len(nuggets))
        final_comment = "Only return the list of labels (List[str]). Do not explain."

        input_sample = {
            "role": "user",
            "content": "{}\nSearch Query: {}\nNugget List: {}\n{}\nLabels:".format(user_instruction, query, str(nuggets), final_comment),
        }
        response = self._openai_client.chat.completions.create(
            model=self._gpt_version,
            messages=prompt + [input_sample],
            max_tokens=max_length,
            temperature=0.0,
        )
        predicted_response = response.choices[0].message.content

        return predicted_response
    
    def assign_nuggets(
        self, query: str, response: str, nuggets: List[str], max_length: int = 1000,
    ) -> List[str]:
        prompt = [
            {
                "role": "system",
                "content": "You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets based on if they are captured by a given passage.",
            },
        ]
        user_instruction = "Based on the query and passage, label each of the {} nuggets either as support, partial_support, or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget is not captured at all, label it as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.".format(len(nuggets))
        final_comment = "Only return the list of labels (List[str]). Do not explain."

        input_sample = {
            "role": "user",
            "content": "{}\nSearch Query: {}\nPassage: {}\nNugget List: {}\n{}\nLabels:".format(user_instruction, query, response, str(nuggets), final_comment),
        }
        response = self._openai_client.chat.completions.create(
            model=self._gpt_version,
            messages=prompt + [input_sample],
            max_tokens=max_length,
            temperature=0.0,
        )
        predicted_response = response.choices[0].message.content

        return predicted_response


class CLAUDE(LLM):
    def __init__(
        self, api_key: str = CLAUDE_API_KEY, claude_version: str = "claude-3-5-haiku-20241022"
    ) -> None:
        """Instantiates an LLM based on OpenAI's GPT model.

        Args:
            api_key: OpenAI API key.
            gpt_version (optional): OpenAI GPT model version. Defaults to
              file-level constant DEFAULT_GPT_VERSION.
        """  # noqa
        self._claude = anthropic.Anthropic(api_key=api_key)
        self._claude_version = claude_version
        

    def detect_nuggets(
        self, query: str, docs: List[str], nuggets: List[str], max_length: int = 1000,
    ) -> List[str]:
        system_content = "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query."
        user_instruction = "Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process). Return only the final list of all nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. Ensure the updated nugget list has at most 30 nuggets (can be less), keeping only the most vital ones. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information."
        final_comment = 'Only update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions). List in the form ["a", "b", ...] and a and b are strings with no mention of "'
        context = ""
        for i, doc in enumerate(docs):
            context = context + "[{}] {}\n".format(i+1, doc)

        user_content = "{}\nSearch Query: {}\nContext: {}Search Query: {}\nInitial Nugget List: {}\nInitial Nugget List Length: {}\n{}\nUpdated Nugget List:".format(user_instruction, query, context, query, str(nuggets), len(nuggets), final_comment)
    
        message = self._claude.messages.create(
            model=self._claude_version,
            max_tokens=max_length,
            temperature=0,
            system=system_content,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_content
                        }
                    ]
                }
            ]
        )
        predicted_response = message.content[0].text

        return predicted_response

    def determine_nuggets_importance(
        self, query: str, nuggets: List[str], max_length: int = 1000,
    ) -> List[str]:
        system_content = "You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query."
        
        user_instruction = "Based on the query, label each of the {} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a “good” answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.".format(len(nuggets))
        final_comment = "Only return the list of labels (List[str]). Do not explain."

        user_content = "{}\nSearch Query: {}\nNugget List: {}\n{}\nLabels:".format(user_instruction, query, str(nuggets), final_comment)
        
        message = self._claude.messages.create(
            model=self._claude_version,
            max_tokens=max_length,
            temperature=0,
            system=system_content,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_content
                        }
                    ]
                }
            ]
        )
        predicted_response = message.content[0].text

        return predicted_response
    
    def assign_nuggets(
        self, query: str, response: str, nuggets: List[str], max_length: int = 1000,
    ) -> List[str]:
        system_content=  "You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets based on if they are captured by a given passage."
        
        user_instruction = "Based on the query and passage, label each of the {} nuggets either as support, partial_support, or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget is not captured at all, label it as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.".format(len(nuggets))
        final_comment = "Only return the list of labels (List[str]). Do not explain."

        user_content = "{}\nSearch Query: {}\nPassage: {}\nNugget List: {}\n{}\nLabels:".format(user_instruction, query, response, str(nuggets), final_comment)
        
        message = self._claude.messages.create(
            model=self._claude_version,
            max_tokens=max_length,
            temperature=0,
            system=system_content,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_content
                        }
                    ]
                }
            ]
        )
        predicted_response = message.content[0].text

        return predicted_response


class GEMINI(LLM):
    def __init__(
        self, system_instruction: str, gemini_version: str = "gemini-1.5-flash"
    ) -> None:
        """Instantiates an LLM based on OpenAI's GPT model.

        Args:
            api_key: OpenAI API key.
            gpt_version (optional): OpenAI GPT model version. Defaults to
              file-level constant DEFAULT_GPT_VERSION.
        """  # noqa
        # system_content = "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query."
        self._gemini = genai.GenerativeModel(model_name=gemini_version, system_instruction=system_instruction)
        self._gemini_version = gemini_version
        

    def detect_nuggets(
        self, query: str, docs: List[str], nuggets: List[str], max_length: int = 1000,
    ) -> List[str]:
        user_instruction = "Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process). Return only the final list of all nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. Ensure the updated nugget list has at most 30 nuggets (can be less), keeping only the most vital ones. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information."
        final_comment = 'Only update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions). List in the form ["a", "b", ...] and a and b are strings with no mention of "'
        context = ""
        for i, doc in enumerate(docs):
            context = context + "[{}] {}\n".format(i+1, doc)

        user_content = "{}\nSearch Query: {}\nContext: {}Search Query: {}\nInitial Nugget List: {}\nInitial Nugget List Length: {}\n{}\nUpdated Nugget List:".format(user_instruction, query, context, query, str(nuggets), len(nuggets), final_comment)
    
        while True:
            try:
                response = self._gemini.generate_content(
                    user_content,
                    generation_config = genai.GenerationConfig(
                        max_output_tokens=max_length,
                        temperature=0.0,
                    )
                )
                predicted_response = response.text
                return predicted_response
            except (google.api_core.exceptions.ResourceExhausted):
                print("Error in response. Trying again.")
                time.sleep(2)
        
    
    def determine_nuggets_importance(
        self, query: str, nuggets: List[str], max_length: int = 1000,
    ) -> List[str]:
        user_instruction = "Based on the query, label each of the {} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a “good” answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.".format(len(nuggets))
        final_comment = "Only return the list of labels (List[str]). Do not explain."

        user_content = "{}\nSearch Query: {}\nNugget List: {}\n{}\nLabels:".format(user_instruction, query, str(nuggets), final_comment)
        
        while True:
            try:
                response = self._gemini.generate_content(
                    user_content,
                    generation_config = genai.GenerationConfig(
                        max_output_tokens=max_length,
                        temperature=0.0,
                    )
                )
                predicted_response = response.text
                return predicted_response
            except (google.api_core.exceptions.ResourceExhausted):
                print("Error in response. Trying again.")
                time.sleep(2)

    
    def assign_nuggets(
        self, query: str, response: str, nuggets: List[str], max_length: int = 1000,
    ) -> List[str]:
        
        user_instruction = "Based on the query and passage, label each of the {} nuggets either as support, partial_support, or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget is not captured at all, label it as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.".format(len(nuggets))
        final_comment = "Only return the list of labels (List[str]). Do not explain."

        user_content= "{}\nSearch Query: {}\nPassage: {}\nNugget List: {}\n{}\nLabels:".format(user_instruction, query, response, str(nuggets), final_comment)
        
        while True:
            try:
                response = self._gemini.generate_content(
                    user_content,
                    generation_config = genai.GenerationConfig(
                        max_output_tokens=max_length,
                        temperature=0.0,
                    )
                )
                predicted_response = response.text
                return predicted_response
            except (google.api_core.exceptions.ResourceExhausted):
                print("Error in response. Trying again.")
                time.sleep(2)