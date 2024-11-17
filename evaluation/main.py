from deepeval.metrics import BiasMetric, AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from typing import List
from pprint import pprint
import pandas as pd

def interpret_results(results):
    success = 0
    score = 0
    model = results[0]["evaluation_model"]
    cost = 0
    for res in results:
      a = list(res.values())
      success += a[-2]
      score += a[-1]
      cost += a[-3]
    return {"Evaluation Model" : model, "Evaluation Cost" : cost, "Evaluation Sucess" : success, 
            "Evaluation Score" : score, "Total Evaluation" : len(results),
            "Avg. Success" : success/len(results), "Avg. Score" : score/len(results)}

def styx_evaluation(df, provider = "deepEval", metric="bias", threshold=0.5):
  """
  The input must be a dataframe and it must have the following columns:
  model_input, response, Expected Output
  """
  test_cases = []
  for _, row in df.iterrows():
    if provider == "deepEval":
      # if "Expected Output" in row:
        # test_cases.append(LLMTestCase(input=row["model_input"], actual_output=row["Expected Output"], context=row["context"]))
        test_cases.append(LLMTestCase(input=row["model_input"], actual_output=row["response"]))
    else:
      if "Expected Output" in row:
        test_cases.append(
          (row["model_input"], row["response"], row["Expected Output"])
        )
      else:
        test_cases.append(
          (row["model_input"], row["response"])
        )
    
  results = []
  checkpoint_file = "checkpoint_eval_results.csv"
  if provider == "deepEval":
    if metric == "bias":
      metric = BiasMetric(threshold=threshold,model='gpt-4o-mini')
    # Check pointing left.
    batch_size = 5
    for i in range(0, len(test_cases), batch_size):
      res = EvaluationDataset(test_cases=test_cases[i:i+batch_size]).evaluate(metrics=[metric]).test_results
      for r in res:
        results.append({
            'prompt' : r.input,
            'response' : r.actual_output,
            'reason': r.metrics_data[0].reason,
            'evaluation_model': r.metrics_data[0].evaluation_model,
            'evaluation_cost': r.metrics_data[0].evaluation_cost,
            'success': r.metrics_data[0].success,
            'score': r.metrics_data[0].score
        })
      
      # Checkpoint and save data after processing each batch
      checkpoint_df = pd.DataFrame(results)
      checkpoint_df.to_csv(checkpoint_file, index=False)
      print(f"Checkpoint saved after processing batch {i // batch_size + 1}")
      
      if i%2 == 0:
        pprint( interpret_results(results) )
    return results
