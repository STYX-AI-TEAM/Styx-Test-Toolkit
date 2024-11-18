from datasets import load_dataset
import pandas as pd
import os
from tqdm import tqdm

def TruthfulQAChatContext(prompt):
    context = []
    prompt = prompt.strip().split("Human: ")
    for p in prompt:
        a = p.strip().split("Assistant: ")
        context.append({"role": "user", "content": a[0]})
        context.append({"role": "assistant", "content": a[1].split("Human: ")[0]})
    return context

def BBQPrompt(prompt, context, options):
    base_prompt = context + prompt + "Options:\n"
    for idx, option in enumerate(options):
        base_prompt += f"{idx+1}. {option}\n"
    base_prompt += """
    You should answer the most likely option in the following xml format:

    ```<option> 1 or 2 or 3... </option>```"""
    return base_prompt

def interpretResponse(response):
    try:
        return int( response.split("<option>")[-1].split("</option>")[0].strip() )
    except:
        for i in response:
            if i.isdigit():
                return int(i)
        return -1  # This is manually done.
    
class StyxDatasets:
    dataset_map = {
        # "CALM": ""  # You may want to populate this with more datasets
        "bold" : "AlexaAI/bold",
        "truthfulQA":"truthfulqa/truthful_qa",
        "calm" : "daishen/CALM-Data",
        "toxigen" : "toxigen/toxigen-data",
        "anthropic" : "nz/anthropic-hh-golden-rlhf",
        "bbq" : "walledai/BBQ",
        "stereoset" : "McGill-NLP/stereoset",
    }
    
    columns_map = {
        # "calm": ["prompt", "output"],
    }
    
    def __init__(self, dataset_name, subset_name=None, split='train', checkpoint_dir='checkpoints', rows = None, dataset_type="chatbot"):
        self.dataset_name = dataset_name
        
        if dataset_name.lower() in self.dataset_map:
            dataset_name = self.dataset_map[dataset_name]
            
        # Load the dataset
        if subset_name:
            data = load_dataset(dataset_name, subset_name)
        else:
            data = load_dataset(dataset_name)
        
        # Load the split into a DataFrame
        if rows is not None:
            self.df = pd.DataFrame(data[split][:rows])
        else:
            self.df = pd.DataFrame(data[split])

        # Identify the columns for this dataset
        # If columns are found, filter the DataFrame to those columns
        cols = None
        for k, v in self.columns_map.items():
            if dataset_name.lower() in k.lower():
                cols = v
                break
        if cols:
            self.df = self.df[cols]

        ###################################
        # Manual Work
        # If dataset is CALM, prepare the "input" column
        if "bold" in dataset_name.lower():
            self.df["model_input"] = self.df["prompts"].apply(lambda x: x[0] if isinstance(x, list) else x)
            self.df["expected_output"] = self.df["wikipedia"].apply(lambda x: x[0] if isinstance(x, list) else x)

        elif "TruthfulQA" in dataset_name.lower():
            self.df["model_input"] = self.df["question"].apply(lambda x: x[0] if isinstance(x, list) else x)
            self.df["expected_output"] = self.df["best_answer"].apply(lambda x: x[0] if isinstance(x, list) else x)

        elif "anthropic" in dataset_name.lower(): 
            if dataset_type == "chatbot":
                self.df["model_input"] = self.df["prompt"].apply(lambda x: x[0] if isinstance(x, list) else x)
                self.df["expected_output"] = self.df["chosen"].apply(lambda x: x[0] if isinstance(x, list) else x)
            else:
                self.df["model_input"] = self.df["prompt"].apply(lambda x: TruthfulQAChatContext(x[0]) if isinstance(x, list) else TruthfulQAChatContext(x))
                self.df["expected_output"] = self.df["chosen"].apply(lambda x: x[0] if isinstance(x, list) else x)

        elif "calm" in dataset_name.lower():
            self.df["model_input"] = self.df["instruction"].apply(lambda x: x[0] if isinstance(x, list) else x)
            self.df["expected_output"] = self.df["output"].apply(lambda x: x[0] if isinstance(x, list) else x)

        elif "toxigen" in dataset_name.lower():
            self.df["model_input"] = self.df["text"].apply(lambda x: x[0] if isinstance(x, list) else x)
            
        elif "bbq" in dataset_name.lower():
            self.df["model_input"] = self.df.apply(
                lambda row: BBQPrompt(row['question'], row['context'], row['choices']),axis=1
            )
            self.df["expected_output"] = self.df["response"].apply(lambda x: x[0] if isinstance(x, list) else x)
        
        # STEREOSET LEFT
        # elif "stereoset" in dataset_name.lower():
        #     self.df["model_input"] = self.df["input"].apply(lambda x: x[0] if isinstance(x, list) else x)
        #     self.df["expected_output"] = self.df["output"].apply(lambda x: x[0] if isinstance(x, list) else x)
        
        else:
            self.df["model_input"] = self.df["input"].apply(lambda x: x[0] if isinstance(x, list) else x)
            self.df["expected_output"] = self.df["output"].apply(lambda x: x[0] if isinstance(x, list) else x)
        
        # Reset index for clean DataFrame
        self.df = self.df.reset_index(drop=True)
        
        # Prepare checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def generate_responses(self, model, max_new_tokens=100, num_return_sequences=1, batch_size=3):
        responses = []
        
        # Try to resume from last checkpoint if it exists
        checkpoint_file = os.path.join(self.checkpoint_dir, 'responses_checkpoint.csv')
        if os.path.exists(checkpoint_file):
            print(f"Resuming from last checkpoint: {checkpoint_file}")
            checkpoint_df = pd.read_csv(checkpoint_file)
            start_idx = len(checkpoint_df)
            responses = checkpoint_df['response'].tolist()
            self.df = self.df.iloc[start_idx:].reset_index(drop=True)
        else:
            start_idx = 0
        
        # Iterate over each prompt to generate a response
        for idx, prompt in tqdm(enumerate(self.df["model_input"]), total=len(self.df), desc="Generating responses"):
            try:
                # Make sure the model is generating a response for each prompt
                response = model.generate(prompt, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences)
                
                if "bbq" in self.dataset_name.lower() or "stereoset" in self.dataset_name.lower():
                    responses.append(interpretResponse(response))
                else:
                    responses.append(response)

                # Periodically save to CSV after processing each batch
                if (idx + 1) % batch_size == 0:
                    self.df.loc[:idx, 'response'] = responses[:idx+1]
                    self.df.to_csv(checkpoint_file, index=False)
                    print(f"Checkpoint saved at batch {idx + 1}")

            except Exception as e:
                print(f"Error generating response for prompt {idx + 1}: {e}")
                break  # Stop and resume from checkpoint in case of an error
        
        # Add all generated responses to the DataFrame
        self.df['response'] = responses

        # Final save after all responses are generated
        self.df.to_csv(checkpoint_file, index=False)
        print(f"Final checkpoint saved at {checkpoint_file}")
        
        return self.df  # Return the DataFrame with the responses
