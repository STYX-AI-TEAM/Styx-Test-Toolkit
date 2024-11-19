import torch, json, requests
from transformers import pipeline

class StyxModels:
  model_map = {
    "gpt 2" : "openai-community/gpt2",
    "llama 3.2" : "meta-llama/Llama-3.2-1B",
    "Gemma 2" : "google/gemma-2-2b-it", 
  }
  model = None
  def __init__(self, model = None, endpoint = None, api_key = None):
    if model is not None:
      device = 0 if torch.cuda.is_available() else -1
      if model in self.model_map:
        model = self.model_map[model]
      self.model = pipeline("text-generation", model=model,device=device)
    self.url = endpoint
    self.headers = {
        "Authorization": f"Bearer {api_key}" if api_key else "",
        "Content-Type": "application/json",
    }
    
  def generate(self, prompt = None, context = None, max_new_tokens = 100, num_return_sequences = 1):
    if self.model is not None:
      return self.model(prompt, max_new_tokens = max_new_tokens, num_return_sequences = num_return_sequences)[-1]['generated_text']
    body = {
            "messages": [{"role": "user", "content": prompt}] if context is None else context,
            "temperature": 0.5,
            "max_tokens": 2500,
            "top_p": 0.7,
            "stream": False
        }
    res = json.loads(requests.post(url=self.url, headers=self.headers, json=body).text)
    return res['choices'][-1]['message']['content']

