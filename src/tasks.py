from src.data import create_prompts
from src.data import create_metaphor_classification_prompt
from src.data import create_source_domain_v2_prompt
from src.data import create_target_domain_v2_prompt
from src.data import create_source_lexeme_prompt
from src.data import create_target_lexeme_prompt
from src.data import prefix_prompts_with_few_shot_examples
from src.models import GPT2, GPT3, Llama2
from src.evaluation import cosine_similarity_for_word_pair
from src.evaluation import compute_scores
from src.utils.workspace import get_workdir
import numpy as np
import pandas as pd
import json
import time
import os
from sklearn.model_selection import ParameterGrid
from typing import Any
from datasets import Dataset

WORKDIR = get_workdir()


def get_few_shot_examples_for_classification(train_df: pd.DataFrame,
                                             sample_values: list[int],
                                             no_of_samples_per_value: int,
                                             random_seed: int) -> list[list[int]]:
    rng = np.random.default_rng(random_seed)
    few_shot_sample_indices = []
    for n_samples in sample_values:
        train_examples = train_df.copy(deep=True)
        for _ in range(no_of_samples_per_value):
            # select positive and negative examples with the same distribution
            positive_examples = list(train_examples[train_examples["label"] == 1].sample(
                n=int(n_samples / 2), random_state=random_seed).index)
            negative_examples = list(train_examples[train_examples["label"] == 0].sample(
                n=int(n_samples / 2), random_state=random_seed).index)
            examples = np.concatenate((positive_examples, negative_examples))
            # shuffle examples
            rng.shuffle(examples)
            examples = examples.tolist()
            few_shot_sample_indices.append(examples)
            # avoid using the same examples for another instance of the same sample value
            train_examples = train_examples.drop(examples)
    return few_shot_sample_indices


def get_few_shot_examples_for_prediction(train_df: pd.DataFrame,
                                         sample_values: list[int],
                                         no_of_samples_per_value: int,
                                         random_seed: int) -> list[list[int]]:
    few_shot_sample_indices = []
    for n_samples in sample_values:
        train_examples = train_df.copy(deep=True)
        for _ in range(no_of_samples_per_value):
            examples = list(train_examples.sample(n=n_samples, random_state=random_seed).index)
            few_shot_sample_indices.append(examples)
            # avoid using the same examples for another instance of the same sample value
            train_examples = train_examples.drop(examples)
    return few_shot_sample_indices


def evaluate_prediction(df_results: pd.DataFrame, task_name: str, model: GPT2 |
                        GPT3 | Llama2) -> tuple[pd.DataFrame, dict[str, Any]]:
    correct = 0
    for i in range(len(df_results)):
        if task_name == "source_domain_prediction":
            gold = df_results.loc[i, 'source_domain']
        elif task_name == "target_domain_prediction":
            gold = df_results.loc[i, 'target_domain']
        elif task_name == "source_lexeme_prediction":
            gold = df_results.loc[i, 'source_lexeme']
        elif task_name == "target_lexeme_prediction":
            gold = df_results.loc[i, 'target_lexeme']
        # elif task_name == "Finetuned":
        #     gold = df_results.loc[i, 'completion'].strip()
        #     gold = gold[:gold.rfind("END")].strip()
        predicted = df_results.loc[i, model.get_completion_key()].strip()
        # check if predicted or gold is multiple words
        if "-" in predicted:
            predicted = predicted.replace("-", " ")
        if predicted.lower() == gold.lower():
            correct += 1
        similarity = cosine_similarity_for_word_pair(predicted, gold)
        df_results.loc[i, 'embedding_fasttext_sim'] = similarity

    acc = correct / len(df_results)
    mean = df_results['embedding_fasttext_sim'].mean()
    std = df_results['embedding_fasttext_sim'].std()
    print(model.get_name() + " Accuracy: ", acc)
    print(model.get_name() + " - (fasttext) Mean similarity: ", mean)
    print(model.get_name() + " - (fasttext) Standard deviation: ", std)
    result = {
        "model": model.get_name(),
        "acc": acc,
        "mean_em": mean,
        "std_em": std,
    }
    return df_results, result


def evaluate_classification(
        df_results: pd.DataFrame,
        task_name: str,
        model: GPT2 | GPT3 | Llama2) -> tuple[pd.DataFrame, dict[str, Any]]:
    for i in range(len(df_results)):
        predicted = df_results.loc[i, model.get_completion_key()]
        predicted = predicted.strip().lower()
        gold = str(df_results.loc[i, 'label'])
        result = 0
        if predicted == gold:
            result = 1
        elif (predicted == "no" and gold == "literal") or (predicted == "no" and gold == "0"):
            result = 1
        elif (predicted == "yes" and gold == "nonliteral") or (predicted == "yes" and gold == "1"):
            result = 1
        df_results.loc[i, 'correct'] = result

    def get_label_val(label: str) -> int:
        label = str(label).strip().lower()
        if label == "nonliteral":
            return 1
        elif label == "literal":
            return 0
        elif label == "yes":
            return 1
        elif label == "no":
            return 0
        elif label == "0" or label == 0:
            return 0
        elif label == "1" or label == 1:
            return 1
        else:
            print(f"Warning: prediction out-of-bounds - '{label}'. Using label '2' to indicate prediction error")
            return 2

    def get_prediction_val(prediction: str, actual: str) -> int:
        pred, act = get_label_val(prediction), get_label_val(actual)
        if pred != 2:
            return pred
        if act == 0:
            return 1
        return 0

    acc = df_results['correct'].sum() / df_results[model.get_completion_key()].count()
    y_true = df_results["label"].apply(get_label_val).to_list()

    y_pred = []
    for i in range(len(df_results["label"])):
        prediction = df_results.at[i, model.get_completion_key()]
        actual = df_results.at[i, "label"]
        y_pred.append(get_prediction_val(prediction, actual))

    f1, precision, recall = compute_scores(y_true, y_pred)
    print(f"Accuracy: {acc}\nF1 score: {f1}\nPrecision: {precision}\nRecall: {recall}\n")

    result = {
        "model": model.get_name(),
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
    return df_results, result


def get_best_validation_result_for_classification(validation_results: list[dict[str, Any]]) -> dict[str, Any]:
    best_idx = np.argmax(
        np.array(list(map(lambda x: x["f1"], validation_results))))
    return validation_results[best_idx]


def get_best_validation_result_for_prediction(validation_results: list[dict[str, Any]]) -> dict[str, Any]:
    best_idx = np.argmax(
        np.array(list(map(lambda x: x["mean_em"], validation_results))))
    return validation_results[best_idx]


class Task:
    def __init__(
            self,
            dataset_name: str,
            task_name: str,
            fine_tuning: bool,
            model: GPT2 | GPT3 | Llama2,
            save_model: bool,
            push_model: bool,
            random_seed: int,
            sample_values: list[int] = [
                2,
                4,
                6,
                8,
                12],
            no_of_samples_per_value: int = 3):
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.fine_tuning = fine_tuning
        self.model = model
        self.save_model = save_model
        self.push_model = push_model
        self.random_seed = random_seed
        self.sample_values = sample_values
        self.no_of_samples_per_value = no_of_samples_per_value
        self.get_few_shot_examples = get_few_shot_examples_for_classification if task_name == "classification" else get_few_shot_examples_for_prediction
        self.evaluate = evaluate_classification if task_name == "classification" else evaluate_prediction
        self.prompt_creator = create_metaphor_classification_prompt
        if task_name == "source_domain_prediction":
            self.prompt_creator = create_source_domain_v2_prompt
        elif task_name == "target_domain_prediction":
            self.prompt_creator = create_target_domain_v2_prompt
        elif task_name == "source_lexeme_prediction":
            self.prompt_creator = create_source_lexeme_prompt
        elif task_name == "target_lexeme_prediction":
            self.prompt_creator = create_target_lexeme_prompt
        self.get_best_validation_result = get_best_validation_result_for_classification if task_name == "classification" else get_best_validation_result_for_prediction

    def get_prompt_completions(self, prompts: list[str], df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        completions = self.model.completion(prompts)
        df_results = df.copy().reset_index()
        model_completion_key = self.model.get_completion_key()
        for i in range(len(completions)):
            df_results.loc[i, model_completion_key] = completions[i].choices[0].text if self.model.get_name(
            ) == "gpt3.5" else completions[i]
        return df_results, completions

    def few_shot_prompting(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> list[dict[str, Any]]:
        print("Lenght of train:", len(train_df))
        print("Lenght of valid:", len(valid_df))
        self.train_prompts = create_prompts(train_df, self.prompt_creator)
        self.valid_prompts = create_prompts(valid_df, self.prompt_creator)

        # Ideally, these few-shot prefix samples should be removed 
        # from the remaining examples, to avoid prompts including 
        # an example prompt + correct answer asking for the same thing.
        few_shot_sample_indices = self.get_few_shot_examples(
            train_df, self.sample_values, self.no_of_samples_per_value, self.random_seed)

        param_grid = {
            'few_shot_sample_indices': few_shot_sample_indices
        }
        param_grid = list(ParameterGrid(param_grid))
        print("Number of parameter combinations:", len(param_grid), "\n")

        os.makedirs(
            f'{WORKDIR}/validation/few_shot/{self.task_name}/{self.dataset_name}/',
            exist_ok=True)
        gpt3_completions = []
        results = []
        for i, params in enumerate(param_grid):
            print("\nCurrent few-shot sample:", i + 1, "/", len(param_grid))
            if i + 1 <= len(param_grid):
                run_name = "val" + "_" + self.model.get_name() + "_temp-" + str(self.model.temperature) + "_" + str(
                    params['few_shot_sample_indices']) + "_(" + str(len(params['few_shot_sample_indices'])) + "-few-shot-samples)"
                prompts = prefix_prompts_with_few_shot_examples(
                    self.train_prompts, self.valid_prompts, params['few_shot_sample_indices'])

                # prompt
                df_results, completions = self.get_prompt_completions(prompts, valid_df)
                if self.model.get_name() == "gpt3.5":
                    gpt3_completions.append(completions)

                # evaluate
                df_results, result = self.evaluate(df_results, self.task_name, self.model)
                result["few_shot_sample_indices"] = params['few_shot_sample_indices']
                result["few_shot_samples"] = len(params['few_shot_sample_indices'])
                result["temperature"] = self.model.temperature
                result["eval_split"] = "val"
                result["dataset_name"] = self.dataset_name
                result["task_name"] = self.task_name
                results.append(result)

                df_results.to_csv(
                    f'{WORKDIR}/validation/few_shot/{self.task_name}/{self.dataset_name}/{run_name}.csv')

        # save gpt3 completions for later cost estimation
        if len(gpt3_completions) > 0:
            os.makedirs(
                f'{WORKDIR}/gpt3_completions/{self.task_name}/{self.dataset_name}/',
                exist_ok=True)
            with open(f'{WORKDIR}/gpt3_completions/{self.task_name}/{self.dataset_name}/{int(time.time())}_gpt3_few-shot-completions.json', "w") as f:
                json.dump(gpt3_completions, f, indent=4)

        with open(f'{WORKDIR}/validation/few_shot/{self.task_name}/{self.dataset_name}/{int(time.time())}_{self.model.get_name()}_prompting_results.json', "w") as f:
            json.dump(results, f, indent=4)

        return results

    def fine_tune(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> str:
        print("Lenght of train:", len(train_df))
        print("Lenght of valid:", len(valid_df))
        self.train_prompts = create_prompts(train_df, self.prompt_creator)
        self.valid_prompts = create_prompts(valid_df, self.prompt_creator)
        train_dataset = []
        for i in range(len(self.train_prompts[0])):
            train_dataset.append(self.train_prompts[0][i] + " " + self.train_prompts[1][i])
        valid_dataset = []
        for i in range(len(self.valid_prompts[0])):
            valid_dataset.append(self.valid_prompts[0][i] + " " + self.valid_prompts[1][i])
        ds_train = Dataset.from_dict({"text": train_dataset})
        ds_validation = Dataset.from_dict({"text": valid_dataset})
        return self.model.fine_tune(
            ds_train,
            ds_validation,
            self.model.get_name() +
            "-" +
            self.dataset_name.replace(
                "_",
                "-") +
            "-" +
            self.task_name.replace(
                "_",
                "-"),
            self.save_model,
            self.push_model)

    def eval(self, test_df: pd.DataFrame, validation_results: list[dict[str, Any]] = None) -> dict[str, Any]:
        print("Lenght of test:", len(test_df))
        test_prompts = create_prompts(test_df, self.prompt_creator)
        if not self.fine_tuning:
            best_config = self.get_best_validation_result(validation_results)
            prompts = prefix_prompts_with_few_shot_examples(
                self.train_prompts, test_prompts, best_config["few_shot_sample_indices"])
        else:
            prompts = test_prompts[0]

        print("Getting completions...")
        df_results, completions = self.get_prompt_completions(prompts, test_df)
        source_completion_prefix = f'{WORKDIR}/test/{"fine_tuning" if self.fine_tuning else "few_shot"}/{self.task_name}/{self.dataset_name}'
        os.makedirs(
            f'{source_completion_prefix}',
            exist_ok=True)
        os.makedirs(f'{WORKDIR}/gpt3_completions/{self.task_name}/{self.dataset_name}', exist_ok=True)
        if self.model.get_name() == "gpt3.5":
            os.makedirs(f'{WORKDIR}/gpt3_completions/{self.task_name}/{self.dataset_name}/', exist_ok=True)
            with open(f'{WORKDIR}/gpt3_completions/{self.task_name}/{self.dataset_name}/{int(time.time())}_gpt3-test-set-completions.json', "w") as f:
                json.dump(completions, f, indent=4)

        # test set eval
        print("Evaluating test set performance...")
        df_results, result = self.evaluate(
            df_results, self.task_name, self.model)
        if not self.fine_tuning:
            result["few_shot_sample_indices"] = best_config["few_shot_sample_indices"]
            result["few_shot_samples"] = best_config["few_shot_samples"]
        result["temperature"] = self.model.temperature
        result["eval_split"] = "test"
        result["dataset_name"] = self.dataset_name
        result["fine_tuning"] = self.fine_tuning
        result["task_name"] = self.task_name

        df_results.to_csv(
            f'{source_completion_prefix}/{int(time.time())}_{self.model.get_name()}_{"best_config_test" if not self.fine_tuning else "test_fine_tuned"}.csv')

        if not self.fine_tuning:
            with open(f'{source_completion_prefix}/{int(time.time())}_{self.model.get_name()}_best_config_test_result.json', "w") as f:
                json.dump(best_config, f, indent=4)

        with open(f'{source_completion_prefix}/{int(time.time())}_{self.model.get_name()}_eval_split_test_result.json', "w") as f:
            json.dump(result, f, indent=4)

        return result
