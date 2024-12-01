from bias.bias import BiasMetric
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

def styx_evaluation(df, provider = "deepEval", metric="bias", threshold=0.5, model = "gpt-4o-mini"):
  """
  The input must be a dataframe and it must have the following columns:
  model_input, response, Expected Output
  """
  # Perpare the test cases
  test_cases = []
  fields = ["model_input", "response"]
  if "expected_output" in df.columns:
    fields.append("expected_output") 
  test_cases = df.apply(lambda row: {field: row[field] for field in fields}, axis=1).tolist()
  
  results = []
  checkpoint_file = "checkpoint_eval_results.csv"
  if provider == "deepEval":
    if metric == "bias":
      metric = BiasMetric(threshold=threshold,model=model)
    for test_case in test_cases:
      for r in metric.measure(test_case):
        results.append({
            'prompt' : test_case['model_input'],
            'response' : test_case['actual_output'],
            'evaluation_model': r.evaluation_model,
            'success': r.success,
            'score': r.score
        })
      
      # Checkpoint and save data after processing each batch
      pd.DataFrame(results).to_csv(checkpoint_file, index=False)
    return results
  else:
    score = sum(df['response'] == df['expected_output'])
    return {'score':score, 'percentage':f'{(score*100)/len(df):.2f}'}