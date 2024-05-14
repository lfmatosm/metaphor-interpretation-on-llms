import openai
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)

import torch
import os
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from src.utils.workspace import get_workdir
openai.api_key = openai.api_key_path = f'{get_workdir()}/GPT3.txt'


class GPT2Dataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

    def collate_fn(self, batch):
        encoding = self.tokenizer.__call__(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length)
        return encoding

# Reference for batch generation with GPT2:
# https://github.com/huggingface/transformers/pull/7552#issue-497255933
class GPT2:
    def __init__(self, fine_tuned_model_path: str = None, temperature: float = 0., max_new_tokens: int = 100):
        self.model_name = "openai-community/gpt2"
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_seq_len = 128

        device_map = {"": 0}

        model_name_or_path = "openai-community/gpt2" if fine_tuned_model_path is None else fine_tuned_model_path

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_name(self) -> str:
        return "gpt2-sm"

    def get_completion_key(self) -> str:
        return "GPT2_SM Completion"

    def completion(self, sentences: list[str]) -> list[str]:
        completions = []

        dt = GPT2Dataset(sentences, self.tokenizer, self.max_seq_len)
        dloader = DataLoader(dt, batch_size=32, collate_fn=dt.collate_fn)

        raw_comps = []

        with torch.no_grad():
            for batch in dloader:
                encoding = {k: v.to(self.model.device) for k, v in batch.items()}
                if self.temperature == 0.:
                    generate_ids = self.model.generate(**encoding, max_new_tokens=self.max_new_tokens, do_sample=False)
                else:
                    generate_ids = self.model.generate(
                        encoding, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
                comps = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                raw_comps.extend(comps)
            for idx, sentence in enumerate(sentences):
                completions.append(raw_comps[idx][len(sentence):])
        return completions

    def fine_tune(self, train_dataset: Dataset, valid_dataset: Dataset, name: str, save: bool, push: bool) -> str:
        training_arguments = TrainingArguments(
            name,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=0,
            logging_steps=25,
            learning_rate=2e-4,
            weight_decay=0.001,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="none",
            push_to_hub=push,
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=self.tokenizer,
            args=training_arguments,
            packing=False,
        )

        trainer.train()

        if save:
            output_dir = f"{name}"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving fine-tuned model to {output_dir}")
            trainer.model.save_pretrained(output_dir)
            trainer.tokenizer.save_pretrained(output_dir)

        if push:
            print(f"Uploading fine-tuned model ({name}) to HuggingFace Hub")
            trainer.push_to_hub()
        return name


class GPT3:
    def __init__(self, fine_tuned_model_path: str = None, temperature: float = 0., max_new_tokens: int = 100):
        self.model_name = "gpt-3.5-turbo-instruct"
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def get_name(self) -> str:
        return "gpt3.5"

    def get_completion_key(self) -> str:
        return "GPT3 Completion"

    # TODO: return type is not string list, but completions object list from openai
    def completion(self, sentences: list[str]) -> list[str]:
        completions = []

        for sentence in sentences:
            c = openai.Completion.create(
                model=self.model_name,
                prompt=sentence,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            completions.append(c)
            # avoiding rate limit
            time.sleep(0.2)
        return completions

    def fine_tune(self, train_dataset: Dataset, valid_dataset: Dataset, name: str, save: bool, push: bool) -> str:
        # TODO: add fine-tune implementation
        return name


class Llama2:
    # TODO: load fine-tuned model
    def __init__(self, fine_tuned_model_path: str = None, temperature: float = 0., max_new_tokens: int = 100):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        #######################################################################
        # bitsandbytes parameters
        #######################################################################
        # Compute dtype for 4-bit base models
        bnb_4bit_compute_dtype = "float16"
        #######################################################################
        # SFT parameters
        #######################################################################
        # Load the entire model on the GPU 0
        device_map = {"": 0}

        # Load tokenizer and model with QLoRA configuration
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            # Activate 4-bit precision base model loading
            load_in_4bit=True,
            # Quantization type (fp4 or nf4)
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            # Activate nested quantization for 4-bit base models (double
            # quantization)
            bnb_4bit_use_double_quant=False,
        )
        # The model that you want to train from the Hugging Face hub
        model_name_or_path = "meta-llama/Llama-2-7b-hf" if fine_tuned_model_path is None else fine_tuned_model_path
        # Load base model
        # Reload model in FP16 and merge it with LoRA weights
        if fine_tuned_model_path is not None:
            base_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                low_cpu_mem_usage=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map=device_map,
            )
            self.model = PeftModel.from_pretrained(base_model, model_name_or_path)
            self.model = self.model.merge_and_unload()
            # Load LLaMA tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                device_map=device_map
            )
            self.model.config.use_cache = False
            self.model.config.pretraining_tp = 1
            # Load LLaMA tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf", trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Fix weird overflow issue with fp16 training
        self.tokenizer.padding_side = "right"
        logging.set_verbosity(logging.CRITICAL)
        if fine_tuned_model_path is None:
            return

        # TODO: uploading takes forever on Colab. To preserve computing resources, we are currently
        # uploading only the adjusted weights and not the entire inference model
        # print(f"Reuploading {fine_tuned_model_path} after merging weights")
        # self.model.push_to_hub(fine_tuned_model_path)
        # self.tokenizer.push_to_hub(fine_tuned_model_path)

    def fine_tune(self, train_dataset: Dataset, valid_dataset: Dataset, name: str, save: bool, push: bool) -> str:
        # Load LoRA configuration
        peft_config = LoraConfig(
            # Alpha parameter for LoRA scaling
            lora_alpha=16,
            # Dropout probability for LoRA layers
            lora_dropout=0.1,
            # LoRA attention dimension
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Set training parameters
        training_arguments = TrainingArguments(
            name,
            # # Output directory where the model predictions and checkpoints will be stored
            # output_dir=output_dir,
            # Number of training epochs
            num_train_epochs=1,
            # Batch size per GPU for training
            per_device_train_batch_size=4,
            # Number of update steps to accumulate the gradients for
            gradient_accumulation_steps=1,
            # Optimizer to use
            optim="paged_adamw_32bit",
            # Save checkpoint every X updates steps
            save_steps=0,
            # Log every X updates steps
            logging_steps=25,
            # Initial learning rate (AdamW optimizer)
            learning_rate=2e-4,
            # Weight decay to apply to all layers except bias/LayerNorm weights
            weight_decay=0.001,
            # Enable fp16/bf16 training (set bf16 to True with an A100)
            fp16=False,
            bf16=False,
            # Maximum gradient normal (gradient clipping)
            max_grad_norm=0.3,
            # Number of training steps (overrides num_train_epochs)
            max_steps=-1,
            # Ratio of steps for a linear warmup (from 0 to learning rate)
            warmup_ratio=0.03,
            # Group sequences into batches with same length
            # Saves memory and speeds up training considerably
            group_by_length=True,
            # Learning rate schedule
            lr_scheduler_type="cosine",
            # Disable external integrations
            report_to="none",
            push_to_hub=push,
        )

        # Set supervised fine-tuning parameters
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            dataset_text_field="text",
            peft_config=peft_config,
            # Maximum sequence length to use
            max_seq_length=None,
            tokenizer=self.tokenizer,
            args=training_arguments,
            # Pack multiple short examples in the same input sequence to increase efficiency
            packing=False,
        )

        # Train model
        trainer.train()

        # Save trained model
        if save:
            output_dir = f"{name}"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving fine-tuned model to {output_dir}")
            trainer.model.save_pretrained(output_dir)

        if push:
            print(f"Uploading fine-tuned model ({name}) to HuggingFace Hub")
            trainer.push_to_hub()
        return name

    def get_name(self) -> str:
        return "llama2-7b"

    def get_completion_key(self) -> str:
        return "LLama2_7b Completion"

    # Batching is problematic for this model as noted in: https://github.com/Stability-AI/lm-evaluation-harness/issues/102
    # As such, batching was reversed to sequential generation for each sentence
    # Better explanation: https://github.com/huggingface/transformers/issues/23017#issuecomment-1649630232
    def completion(self, sentences: list[str]) -> list[str]:
        completions = []
        with torch.no_grad():
            for sentence in sentences:
                inputs = self.tokenizer(sentence, return_tensors="pt").to(self.model.device)
                if self.temperature == 0.:
                    generate_ids = self.model.generate(
                        inputs.input_ids, max_new_tokens=self.max_new_tokens, do_sample=False)
                else:
                    generate_ids = self.model.generate(
                        inputs.input_ids, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
                # decodes output sequence for sentence
                sentence_completion = self.tokenizer.batch_decode(
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                # remove prompt prefix, leaving just completion string
                completions.append(sentence_completion[len(sentence):])
        return completions
