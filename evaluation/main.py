from deepeval.metrics import BiasMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase
from typing import List

def styx_evaluation(df, provider = "deepEval", metric="bias", threshold=0.5):
  test_cases = []
  for _, row in df.iterrows():
    if provider == "deepEval":
      # if "Expected Output" in row:
      #   test_cases.append(LLMTestCase(input=row["model_input"], actual_output=row["Expected Output"], context=row["context"]))
        test_cases.append(LLMTestCase(input=row["model_input"], actual_output=row["response"]))
    test_cases.append(
      (row["model_input"], row["response"], 
       row["Expected Output"] if "Expected Output" in row else row["output"])
    )
    
  if provider == "deepEval":
    if metric == "bias":
      metric = BiasMetric(threshold=threshold,model='gpt-4o-mini')
    # Check pointing left.
    dataset = EvaluationDataset(test_cases=test_cases, metric=[metric])
    return dataset