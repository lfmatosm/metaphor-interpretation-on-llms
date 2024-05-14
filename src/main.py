import argparse
import json
import torch
from src.models import GPT2, GPT3, Llama2
from src.data import load_splits
from src.tasks import Task
from huggingface_hub import interpreter_login
interpreter_login(new_session=False)


def main(
        llm: str,
        dataset: str,
        task_name: str,
        save_model: bool,
        push_model: bool,
        temperature: float = 0.,
        seed: int = 1,
        evaluate: bool = False,
        fine_tuning: bool = False,
        fine_tuned_model_path: str = None):
    if task_name == "classification":
        assert (dataset == "trofi" or dataset == "vua_pos" or dataset ==
                "vua_verb"), "classification task is only available for trofi, vua_pos or vua_verb datasets"
    elif task_name == "source_domain_prediction" or task_name == "target_domain_prediction":
        assert (dataset == "lcc_en_subset" or dataset ==
                "metaphor_list"), "source_domain_prediction task is only available for lcc_en_subset or metaphor_list datasets"
    else:
        assert (dataset == "lcc_en_subset"), f"{task_name} task is only avaiable for lcc_en_subset dataset"

    if fine_tuned_model_path is not None:
        assert (llm == "llama2-7b" or llm == "gpt2-sm"), "fine-tuned model loading is only available for hf models"

    def get_model(llm: str) -> GPT2 | GPT3 | Llama2:
        if llm == "gpt2-sm":
            return GPT2
        elif llm == "gpt3.5":
            return GPT3
        else:
            return Llama2

    (train_df, test_df, valid_df) = load_splits(dataset)
    max_new_tokens = 1 if task_name == "classification" else 10
    if not fine_tuning and fine_tuned_model_path is None:
        model = get_model(llm)(temperature=temperature, max_new_tokens=max_new_tokens)
        task = Task(dataset, task_name, fine_tuning, model, save_model, push_model, seed)

        valid_results = task.few_shot_prompting(train_df, valid_df)
        print(f'Validation results: {json.dumps(valid_results, indent = 4)}\n')
        if not evaluate:
            return

        test_results = task.eval(test_df, valid_results)
        print(f'Test results: {json.dumps(test_results, indent = 4)}\n')
    else:
        if fine_tuning and fine_tuned_model_path is None:
            model = get_model(llm)(temperature=temperature, max_new_tokens=max_new_tokens)
            task = Task(dataset, task_name, fine_tuning, model, save_model, push_model, seed)

            output_path = task.fine_tune(train_df, valid_df)
            print(f"Fine-tuned model saved to '{output_path}'")

        if not evaluate:
            return

        if fine_tuned_model_path is not None:
            print("Fine-tuned model path received, model will be loaded for evaluation")
            output_path = fine_tuned_model_path

        # Empty VRAM if needed
        if 'task' in locals():
            del task

        if 'model' in locals():
            del model

        import gc
        print("No. of unreachable objects:", gc.collect())
        torch.cuda.empty_cache()

        model = get_model(llm)(fine_tuned_model_path=output_path, temperature=temperature, max_new_tokens=max_new_tokens)
        task = Task(dataset, task_name, fine_tuning, model, save_model, push_model, seed)
        valid_results = task.eval(valid_df)
        print(f"Validation results: {json.dumps(valid_results, indent = 4)}\n")
        test_results = task.eval(test_df)
        print(f"Test results: {json.dumps(test_results, indent = 4)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="few-shot prompt tuning or fine-tuning for multiple metaphor datasets")
    parser.add_argument(
        "--llm",
        metavar="LLM",
        type=str,
        choices=[
            "llama2-7b",
            "gpt2-sm",
            "gpt3.5"],
        help="LLM architecture, llama2-7b, gpt2-sm, gpt3.5",
        required=True)
    parser.add_argument(
        "--fine_tuned_model_path",
        metavar="P",
        type=str,
        help="path to fine-tuned model. Only available for hf models"
    )
    parser.add_argument(
        "--dataset",
        metavar="D",
        type=str,
        choices=[
            "metaphor_list",
            "trofi", "lcc_en_subset", "vua_pos", "vua_verb"],
        required=True)
    parser.add_argument(
        "--task",
        metavar="TASK",
        type=str,
        choices=[
            "classification",
            "source_domain_prediction",
            "target_domain_prediction",
            "source_lexeme_prediction",
            "target_lexeme_prediction"],
        required=True)
    parser.add_argument(
        "--temperature",
        metavar="TEMP",
        type=float,
        default=0.,
        help="temperature of the model, controls the level of unpredictability. 0 for a more deterministic behavior (greedy decoding)")
    parser.add_argument(
        "--seed",
        metavar="S",
        type=int,
        default=1,
        help="random seed for reproducibility"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="wheter to evaluate results with test dataset")
    parser.add_argument(
        "--fine_tuning",
        action="store_true",
        help="wheter to fine-tune the given model on the given dataset and task or to use fine-tuned model for prompting. If False, run few-shot prompting by default",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="wheter to save fine-tuned model locally"
    )
    parser.add_argument(
        "--push-model",
        action="store_true",
        default=False,
        help="wheter to push fine-tuned model to HuggingFace Hub"
    )
    args = parser.parse_args()
    print(f"\nllm: {args.llm}\nfine_tuned_model_path: {args.fine_tuned_model_path}\ndataset: {args.dataset}\ntask: {args.task}\ntemperature: {args.temperature}\nseed: {args.seed}\nevaluate: {args.evaluate}\nfine_tuning: {args.fine_tuning}\n\n")
    main(
        args.llm,
        args.dataset,
        args.task,
        args.save_model,
        args.push_model,
        args.temperature,
        args.seed,
        args.evaluate,
        args.fine_tuning,
        args.fine_tuned_model_path)
