import json
from tqdm import tqdm
from typing import Tuple, List, Dict
from datasets import Dataset, load_dataset
from transformers import pipeline, Pipeline, AutoTokenizer
import evaluate
import argparse
import time

HF_DATASET_PATH = "anaterna/airflow-class-summarization"
MAX_NEW_TOKENS = 256

def load_data(dataset_path: str, test_size: float = 0.2, seed: int = 42) -> Dataset:
    """
    Load and split the dataset into training and evaluation sets.
    """
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path, split="train")
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    eval_dataset = split_dataset["test"]
    print(f"Loaded {len(eval_dataset)} samples for evaluation.")
    return eval_dataset

def load_pipeline_and_tokenizer(model_name: str) -> Tuple[Pipeline, AutoTokenizer]:
    """
    Load a text-generation pipeline and its tokenizer.
    """
    print(f"Loading pipeline for model: {model_name}")
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Pipeline and tokenizer loaded.")
    return pipe, tokenizer

def format_prompt(source_code: str, class_name: str) -> str:
    """
    Format the prompt to match the training format.
    """
    return (
        f"{source_code}\n\n"
        '"""Docstring:\n'
    )

def run_evaluation(
    eval_dataset: Dataset,
    pipe: Pipeline,
    max_new_tokens: int,
    model_alias: str
) -> Tuple[List[str], List[str]]:
    """
    Generate summaries for the evaluation dataset and report token lengths.
    """
    references: List[str] = []
    predictions: List[str] = []

    print(f"Starting evaluation for model: {model_alias}")
    for sample in tqdm(eval_dataset, desc=f"Evaluating [{model_alias}]"):
        source_code = sample["source_code"]
        ref_docstring = sample["docstring"]
        class_name = sample["class_name"]

        prompt = format_prompt(source_code, class_name)
        
        output = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,            
            top_p=0.9,                  
            temperature=0.7
        )
        pred_docstring = output[0]["generated_text"].split("Docstring:")[-1].strip()
        word_count = len(pred_docstring.split())
        
        if word_count >= 5:
            print("Initial generation successful.")
        else:
            retries = 4
            attempt = 0

            while attempt < retries:
                output = pipe(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,            
                    top_p=0.9,                  
                    temperature=0.7
                )
                pred_docstring = output[0]["generated_text"].split("Docstring:")[-1].strip()
                word_count = len(pred_docstring.split())
                time.sleep(5)

                if word_count >= 5:
                    break

                attempt += 1

        references.append(ref_docstring)
        predictions.append(pred_docstring)

    return references, predictions

def compute_metrics(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """
    Compute chrF and ROUGE-L metrics.
    """
    print("Computing chrF and ROUGE-L metrics...")
    chrf = evaluate.load("chrf")
    chrf_result = chrf.compute(predictions=predictions, references=references)
    chrf_score = chrf_result['score']

    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rougeL"]
    )
    rouge_l = rouge_result["rougeL"]
    rouge_f1 = rouge_l if isinstance(rouge_l, float) else rouge_l.mid.fmeasure

    print(f"chrF: {chrf_score:.4f} | ROUGE-L (F1): {rouge_f1:.4f}")

    return {
        "chrF": chrf_score,
        "rougeL": rouge_f1
    }

def save_metrics(scores: Dict[str, float], output_file: str) -> None:
    """
    Save evaluation metrics to a JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"Metrics saved to {output_file}")

def save_predictions(references: List[str], predictions: List[str], output_file: str) -> None:
    """
    Save references and predictions to a JSON file (as a list of dicts).
    """
    results = []
    for ref, pred in zip(references, predictions):
        results.append({"reference": ref, "prediction": pred})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed predictions saved to {output_file}")

def main() -> None:
    finetuned_model = "<>"
    original_model = "meta-llama/Llama-3.2-1B"

    parser = argparse.ArgumentParser(description="Evaluate two code summarization models.")
    parser.add_argument("--model1", required=False, help="Hugging Face model checkpoint for model 1.", default=original_model)
    parser.add_argument("--alias1", required=False, help="Alias/tag for model 1 (used in filenames).", default="llama-original")
    parser.add_argument("--model2", required=False, help="Hugging Face model checkpoint for model 2.", default=finetuned_model)
    parser.add_argument("--alias2", required=False, help="Alias/tag for model 2 (used in filenames).", default="llama-finetuned")

    args = parser.parse_args()

    eval_dataset = load_data(HF_DATASET_PATH)

    for model_name, alias in [(args.model1, args.alias1)]:
        pipe, _ = load_pipeline_and_tokenizer(model_name)

        references, predictions = run_evaluation(
            eval_dataset,
            pipe,
            MAX_NEW_TOKENS,
            model_alias=alias
        )

        scores = compute_metrics(references, predictions)

        metrics_file = f"evaluation_results_{alias}.json"
        predictions_file = f"detailed_predictions_{alias}.json"

        save_metrics(scores, metrics_file)
        save_predictions(references, predictions, predictions_file)

if __name__ == "__main__":
    main()
