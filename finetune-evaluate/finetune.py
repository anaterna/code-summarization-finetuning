from typing import Dict, Any
import torch
import transformers
import datasets
import peft
import evaluate


class LlamaDocstringFineTuner:
    def __init__(
        self,
        repo_id: str,
        base_model: str = "meta-llama/Llama-3.2-1B",
        dataset_path: str = "anaterna/airflow-class-summarization",
        lr: float = 2e-4,
        batch_size: int = 1,
        epochs: int = 3,
        seed: int = 42,
    ) -> None:
        self.repo_id = repo_id
        self.base_model = base_model
        self.dataset_path = dataset_path
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_model, use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model and prepare LoRA adapters
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model
        ).to("mps")

        lora_config = peft.LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = peft.get_peft_model(self.model, lora_config)

    def _format(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Formats each data example into prompt + target for fine-tuning."""
        max_length = 512
        source_code = example['source_code'].strip()
        code_part = f"{source_code}\n\n"
        instruction = "Docstring:\n"

        # Dynamically calculate how much room is left for code
        instruction_len = len(self.tokenizer(instruction, add_special_tokens=False)["input_ids"])
        max_code_len = max_length - instruction_len

        # Tokenize and truncate code
        tokenized_code = self.tokenizer(
            code_part,
            truncation=True,
            padding=False,
            max_length=max_code_len
        )
        prompt = self.tokenizer.decode(tokenized_code["input_ids"]) + instruction

        # Final tokenized inputs
        tokenized_input = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        tokenized_labels = self.tokenizer(
            example["docstring"] + self.tokenizer.eos_token,
            truncation=True,
            padding="max_length",
            max_length=256
        )

        return {
            "input_ids": tokenized_input.input_ids,
            "attention_mask": tokenized_input.attention_mask,
            "labels": tokenized_labels.input_ids,
        }

    def load_data(self) -> None:
        """Loads and processes the dataset."""
        ds = datasets.load_dataset(self.dataset_path, split="train")
        split = ds.train_test_split(test_size=0.2, seed=self.seed)
        self.train_set = split["train"].map(self._format, remove_columns=ds.column_names)
        self.eval_set = split["test"].map(self._format, remove_columns=ds.column_names)

    def compute_metrics(self, eval_preds: Any) -> Dict[str, float]:
        """Computes chrF and ROUGE-L metrics for evaluation."""
        preds, labels = eval_preds

        # Handle logits if present
        if isinstance(preds, tuple):
            preds = preds[0]

        # Convert logits or IDs to token IDs
        if isinstance(preds, torch.Tensor):
            pred_ids = torch.argmax(preds, dim=-1)
        else:
            pred_ids = torch.tensor(preds).argmax(dim=-1)

        preds_text = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        labels = torch.tensor(labels)
        labels[labels == -100] = self.tokenizer.pad_token_id
        labels_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute metrics
        chrf_score = evaluate.load("chrf").compute(
            predictions=preds_text, references=labels_text
        )
        rouge_score = evaluate.load("rouge").compute(
            predictions=preds_text, references=labels_text, rouge_types=["rougeL"]
        )
        print(f"chrF: {chrf_score}, ROUGE-L: {rouge_score}")
        return {"chrF": chrf_score["score"], "ROUGE-L": rouge_score["rougeL"]}

    def train(self) -> None:
        """Fine-tunes the model and pushes it to Hugging Face Hub."""
        data_collator = transformers.DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=8,
            learning_rate=self.lr,
            num_train_epochs=self.epochs,
            logging_steps=5,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            output_dir="finetune_tmp",
            report_to="none",
            seed=self.seed,
            label_names=["labels"],
        )

        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_set,
            eval_dataset=self.eval_set,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

        # Save locally
        save_dir = "final_model"
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Final model saved to {save_dir}")

        # Push to Hub
        self.model.push_to_hub(self.repo_id)
        self.tokenizer.push_to_hub(self.repo_id)
        print(f"Model pushed to Hugging Face Hub: {self.repo_id}")


if __name__ == "__main__":
    tuner = LlamaDocstringFineTuner(repo_id="<>")
    tuner.load_data()
    tuner.train()
