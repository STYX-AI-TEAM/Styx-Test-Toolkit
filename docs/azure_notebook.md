# About
This is about using the StyxTest toolkit in azure notebooks

# Installations
Install the following requirements
```
git clone https://github.com/STYX-AI-TEAM/StyxTest
pip install deepeval transformers openai requests datasets
```

# Imports
```
from openai import OpenAI

from time import time
import os

# StyxTest
from StyxTest.models.main import StyxModels
from StyxTest.dataset.main import StyxDatasets
from StyxTest.evaluation.main import styx_evaluation
```

# Usage
### Loading the model
```
model = StyxModels(model="gemma")
```

You can view the available models by,
```
StyxModels().model_map
```

### Loading the dataset
```
dataset=StyxDatasets("bbq",split='age',rows=20)
```

You can view the available datasets by,
```
StyxDatasets("calm").datasets_map
```

### Generating Responses
```
dataset.generate_responses(model)
```

### Evaluating Responses
```
styx_evaluation(dataset.df, provider="custom")
```

# Additional Features
### Checkpointing (Optional)
Since the response generation is an expensive step, keeping track of previous responses is a must. Checkpointing will help us achieving the same.
If you feel it as unnecessary, you can disable it by,
```
dataset.generate_responses(model, checkpointing = False)
```
### Checkpointing Filename (Optional)
When executing with multiple notebooks, having a same resource may cause trouble with same filename for checkpointing. So you can have your own filename as,
```
dataset.generate_responses(model, filename = 'responses.csv')
```
### Custom LLM Judge
For providers like `deepeval`, you can have your custom llm as judge by,
```
from StyxTest.models.llm_judge import CustomJudge

llama_judge = CustomJudge(name="llama", endpoint = "", api_key = "")

# Use it in testing as follows
styx_evaluation(dataset.df, provider="deepeval", model=llama_judge)
```