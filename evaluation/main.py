def generalized_evaluation(
    csv_path, 
    input_column, 
    response_column, 
    context_column=None, 
    metric=None, 
    threshold=0.5
):
  
  data = pd.read_csv(csv_path)
  
  test_cases = []
  for _, row in data.iterrows():
    input_text = row[input_column]
    generated_response = row[response_column]
    context = row[context_column] if context_column else None
