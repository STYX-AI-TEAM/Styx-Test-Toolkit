from datasets import load_dataset
import pandas as pd
import os
from tqdm import tqdm

class StyxDatasets:
    dataset_map = {
        # "CALM": ""  # You may want to populate this with more datasets
        "bold" : "AlexaAI/bold",
    }
    
    columns_map = {
        # "calm": ["prompt", "output"],
    }
    
    def __init__(self, dataset_name, subset_name=None, split='train', checkpoint_dir='checkpoints', rows = None):
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
            self.df["model_input"] = self.df["prompts"]
            self.df["expected_output"] = self.df["wikipedia"]
        
        # Reset index for clean DataFrame
        self.df = self.df.reset_index(drop=True)
        
        # Prepare checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def generate_responses(self, model, max_new_tokens=100, num_return_sequences=1, batch_size=3, **kwargs):
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
                response = model.generate(prompt, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, **kwargs)
                responses.append(response[0])  # Assuming the model returns a list and we take the first one

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