from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import pandas as pd

def load_transformer_model(model_path):
    """
    Load a transformer model and tokenizer from a given directory.

    :param model_path: Path to the directory containing model files.
    :return: A tuple of (model, tokenizer).
    """
    # Load pre-trained model
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    
    # Load pre-trained model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer




def preprocess_tydiqa_dataset(language, tokenizer, dataset_subset=1.0):
    # Load the dataset
    tydiqa_dataset = load_dataset('copenlu/answerable_tydiqa')

    # Filter the dataset for the specified language
    train_dataset = tydiqa_dataset["train"].filter(lambda example: example['language'] == language)
    val_dataset = tydiqa_dataset["validation"].filter(lambda example: example['language'] == language)

    # Sample a subset of the dataset
    train_dataset = train_dataset.shuffle(seed=42).select(range(int(len(train_dataset) * dataset_subset)))
    val_dataset = val_dataset.shuffle(seed=42).select(range(int(len(val_dataset) * dataset_subset)))

    def preprocess_function(examples):
        # Tokenize the examples
        tokenized_inputs = tokenizer(
            examples['question_text'],
            examples['document_plaintext'],
            truncation="only_second",
            max_length=512,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        # Extract overflow_to_sample_mapping and remove it from tokenized_inputs
        overflow_to_sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
        offset_mappings = tokenized_inputs.pop("offset_mapping")

        # Initialize new lists for storing outputs
        start_positions = []
        end_positions = []
        answer_texts = []
        
        # Iterate through the annotations and calculate start and end token positions
        for i, offsets in enumerate(offset_mappings):
            parent_id = overflow_to_sample_mapping[i]
            answer_start = examples['annotations'][parent_id]['answer_start'][0]
            answer_text = examples['annotations'][parent_id]['answer_text'][0]
            answer_end = answer_start + len(answer_text)

            # Find the start and end token index for the answer
            start_token_idx = end_token_idx = 0
            for idx, (start, end) in enumerate(offsets):
                if start <= answer_start < end:
                    start_token_idx = idx
                if start < answer_end <= end:
                    end_token_idx = idx
                    break

            start_positions.append(start_token_idx)
            end_positions.append(end_token_idx)
            answer_texts.append(answer_text)

        # Return the new lists as a dictionary
        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'start_positions': start_positions,
            'end_positions': end_positions,
            'answer_texts': answer_texts
        }

    # Preprocess the datasets
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

    return train_dataset, val_dataset



def zero_shot_eval(model, device, language, tokenizer, modellang):


    train_dataset, val_dataset = preprocess_tydiqa_dataset(language, tokenizer)

    # Collate function to prepare data batches
    data_collator = DefaultDataCollator(return_tensors="pt")

    # DataLoader for validation set
    val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=data_collator)

    
    
    total_loss = 0.0  # Variable to store the total loss
    all_preds_start, all_preds_end, all_true_start, all_true_end = [], [], [], []
    model.eval()
    with torch.no_grad():
        
        
        for batch in val_dataloader:
            # Move batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)

            # Compute the loss (assuming you have a 'loss' key in your outputs)
            loss = outputs.loss

            # Update total loss
            total_loss += loss.item()

            # Get predicted start and end positions
            preds_start = torch.argmax(outputs.start_logits, dim=1)
            preds_end = torch.argmax(outputs.end_logits, dim=1)

            # Get true start and end positions
            true_start = batch["start_positions"]
            true_end = batch["end_positions"]

            # Append predictions and true values for accuracy calculation
            all_preds_start.extend(preds_start.cpu().tolist())
            all_preds_end.extend(preds_end.cpu().tolist())
            all_true_start.extend(true_start.cpu().tolist())
            all_true_end.extend(true_end.cpu().tolist())

    # Calculate average loss
    average_loss = total_loss / len(val_dataloader)
    print(f"{modellang} average Loss: {average_loss:.4f}")

    # Calculate accuracy
    accuracy_start = accuracy_score(all_true_start, all_preds_start)
    accuracy_end = accuracy_score(all_true_end, all_preds_end)
    return_string = f"{language} -> {modellang}"
    print(return_string)
    print(f"{modellang} accuracy (Start): {accuracy_start:.4f}")
    print(f"{modellang} accuracy (End): {accuracy_end:.4f}")

    result_df = pd.DataFrame({
    "Average Loss": [average_loss],
    "Accuracy (Start)": [accuracy_start],
    "Accuracy (End)": [accuracy_end]
    })

    return return_string, result_df
