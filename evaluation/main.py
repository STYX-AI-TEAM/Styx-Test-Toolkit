from deepeval.metrics import BiasMetric
from deepeval.dataset import EvaluationDataset

def styx_evaluation(df, provider = "deepEval", metric="bias", threshold=0.5):
  test_cases = []
  for _, row in df.iterrows():
    test_cases.append(
      (row["model_input"], row["response"], 
       row["Expected Output"] if "Expected Output" in row else row["output"])
    )

  if provider == "deepEval":
    pass