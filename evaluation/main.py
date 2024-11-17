def generalized_evaluation(df, metric=None, threshold=0.5):
  test_cases = []
  for _, row in df.iterrows():
    test_cases.append((row["model_input"], row["response"], row["Expected Output"] if "Expected Output" in row else row["output"]))
