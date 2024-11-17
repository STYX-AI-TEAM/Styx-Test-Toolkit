# About
This folder contains the code to load various models.

# How to use
- Check whether your model exists in the available models or not by
  ```
  model = StyxTest().models
  ```
- Incase of custom models you can either use API endpoints as
  ```
  model = StyxTest().load_api_model(apiendpoint, apikey)
  ```
  or use huggingface models as
  ```
  model = StyxTest().load_hf_model(modelname, num_responses, return_type, ... )
  ```
- Then you can get the responses as
  ```
  model.generate()
  ```
