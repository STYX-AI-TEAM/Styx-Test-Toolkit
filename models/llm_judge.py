import json, requests

from pydantic import BaseModel

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os

class CustomJudge():
    model_name_map = {
        "llama" : "llama 3.1 70B",
        "gpt" : "gpt-4o-mini",
        "openai" : "gpt-4o-mini"
    }
    def __init__(self, name = "llama", endpoint = None, api_key = None):
        model_name = name.lower() if name.lower() in self.model_name_map else "gpt-4o-mini"
        # Messy and un intended code
        if model_name == "gpt":
            model_name = "openai"
        self.name = self.model_name_map[model_name]
        self.endpoint = endpoint if endpoint is not None else os.getenv(f"{model_name.upper()}_JUDGE_ENDPOINT")
        self.api_key = api_key if api_key is not None else os.getenv(f"{model_name.upper()}_JUDGE_API_KEY")
        print(f"Using model: {self.name}")
        
    def load_model(self):
        return self

    def generate(self, prompt: str, schema: BaseModel | None = None) -> BaseModel:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }
        data = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 2500,
            "top_p": 0.7,
            "stream": False,
        }
        response = requests.post(self.endpoint, headers=headers, json=data).json()['choices'][-1]['message']['content']
        try:
            return response if schema is None else schema(**json.loads(response))
        except Exception as e:
            return response if schema is None else schema(**json.loads(response.split("```")[1])) # As the response is in this JSON format

    async def a_generate(self, prompt: str, schema: BaseModel | None = None) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.name