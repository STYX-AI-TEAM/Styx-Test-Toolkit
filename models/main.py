import torch, requests
from transformers import pipeline

class StyxModels:
  model_map = {
    "gpt 2" : "openai-community/gpt2",
    "llama 3.2" : "meta-llama/Llama-3.2-1B",
    "gemma 2" : "google/gemma-2-2b-it",
    "gemma" : "google/gemma-2-2b-it"
  }
  model = None
  def __init__(self, model = None, endpoint = None, api_key = None):
    if endpoint is None and (model is not None):
      device = 0 if torch.cuda.is_available() else -1
      model = self.model_map.get(model.lower(), model)
      print("Loading Model: ", model)
      self.model = pipeline("text-generation", model=model,device=device)
    self.url = endpoint
    self.headers = {
        "Authorization": f"Bearer {api_key}" if api_key else "",
        "api-key": api_key if api_key else "",
        "Content-Type": "application/json",
    }
    
  def generate(self, prompt = None, context = None, max_new_tokens = 100, num_return_sequences = 1):
    if self.model is not None:
      return self.model(prompt, max_new_tokens = max_new_tokens, num_return_sequences = num_return_sequences)[0]['generated_text'][len(prompt):].strip()
    body = {
            "messages": [{"role": "user", "content": prompt}] if context is None else context,
            "temperature": 0.5,
            "max_tokens": max_new_tokens,
            "top_p": 0.7,
            "stream": False
        }
    try:
      res = requests.post(url=self.url, headers=self.headers, json=body).json()
      return res['choices'][0]['message']['content']
    except Exception as e:
      print(e)
      print("Error Response: ", res)
      return None

