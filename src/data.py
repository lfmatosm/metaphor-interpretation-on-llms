import pandas as pd
from src.utils.workspace import get_workdir
from typing import Callable

WORKDIR = get_workdir()


def load_splits(
        dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if dataset_name == "metaphor_list":
        train_df = pd.read_csv(
            f'{WORKDIR}/Data/metaphor_list_with_vua_literal/train.csv', sep=";")
        train_df["source_domain"] = train_df["Source Domain"]
        train_df["target_domain"] = train_df["Target Domain"]
        train_df["example"] = train_df["Example"]
        train_df = train_df.drop(columns=["Source Domain", "Target Domain", "Example"])
        test_df = pd.read_csv(
            f'{WORKDIR}/Data/metaphor_list_with_vua_literal/test.csv', sep=";")
        test_df["source_domain"] = test_df["Source Domain"]
        test_df["target_domain"] = test_df["Target Domain"]
        test_df["example"] = test_df["Example"]
        test_df = test_df.drop(columns=["Source Domain", "Target Domain", "Example"])
        dev_df = pd.read_csv(
            f'{WORKDIR}/Data/metaphor_list_with_vua_literal/dev.csv',
            sep=";")
        dev_df["source_domain"] = dev_df["Source Domain"]
        dev_df["target_domain"] = dev_df["Target Domain"]
        dev_df["example"] = dev_df["Example"]
        dev_df = dev_df.drop(columns=["Source Domain", "Target Domain", "Example"])
    elif dataset_name == "trofi":
        def normalize_label(label: str) -> int:
            if label == "nonliteral":
                return 1
            return 0

        train_df = pd.read_csv(
            f"{WORKDIR}/Data/metaphors_in_plm/trofi/train.csv")
        train_df["label"] = train_df["label"].apply(normalize_label)
        test_df = pd.read_csv(
            f"{WORKDIR}/Data/metaphors_in_plm/trofi/test.csv")
        test_df["label"] = test_df["label"].apply(normalize_label)
        dev_df = pd.read_csv(f"{WORKDIR}/Data/metaphors_in_plm/trofi/dev.csv")
        dev_df["label"] = dev_df["label"].apply(normalize_label)
    elif dataset_name == "vua_pos" or dataset_name == "vua_verb":
        train_df = pd.read_csv(f"{WORKDIR}/Data/metaphors_in_plm/{dataset_name}_subset/train.csv")
        test_df = pd.read_csv(f"{WORKDIR}/Data/metaphors_in_plm/{dataset_name}_subset/test.csv")
        dev_df = pd.read_csv(f"{WORKDIR}/Data/metaphors_in_plm/{dataset_name}_subset/dev.csv")
    elif dataset_name == "lcc_en_subset":
        def normalize_domain(domain: str) -> str:
            return " ".join(domain.strip().lower().split("_"))

        train_df = pd.read_csv(f"{WORKDIR}/Data/lcc_en_subset/train.csv", sep=";")
        train_df["source_domain"] = train_df["source_domain"].apply(normalize_domain)
        train_df["target_domain"] = train_df["target_domain"].apply(normalize_domain)
        train_df["example"] = train_df["sentence"]
        train_df = train_df.drop(columns=["sentence"])
        test_df = pd.read_csv(f"{WORKDIR}/Data/lcc_en_subset/test.csv", sep=";")
        test_df["source_domain"] = test_df["source_domain"].apply(normalize_domain)
        test_df["target_domain"] = test_df["target_domain"].apply(normalize_domain)
        test_df["example"] = test_df["sentence"]
        test_df = test_df.drop(columns=["sentence"])
        dev_df = pd.read_csv(f"{WORKDIR}/Data/lcc_en_subset/dev.csv", sep=";")
        dev_df["source_domain"] = dev_df["source_domain"].apply(normalize_domain)
        dev_df["target_domain"] = dev_df["target_domain"].apply(normalize_domain)
        dev_df["example"] = dev_df["sentence"]
        dev_df = dev_df.drop(columns=["sentence"])
    else:
        raise Exception(f"Unknown dataset split: '{dataset_name}'")
    return train_df, test_df, dev_df


def create_source_domain_prompts(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        use_v2: bool = False):
    train_prompts = []
    train_completions = []
    test_prompts = []
    test_completions = []
    valid_prompts = []
    valid_completions = []

    create_source_domain_prompt_fn = create_source_domain_prompt if not use_v2 else create_source_domain_v2_prompt

    for _, row in train_df.iterrows():
        train_example, completion_example = create_source_domain_prompt_fn(row)
        train_prompts.append(train_example)
        train_completions.append(completion_example)

    for _, row in test_df.iterrows():
        test_example, completion_example = create_source_domain_prompt_fn(row)
        test_prompts.append(test_example)
        test_completions.append(completion_example)

    for _, row in valid_df.iterrows():
        valid_example, completion_example = create_source_domain_prompt_fn(row)
        valid_prompts.append(valid_example)
        valid_completions.append(completion_example)

    return [
        train_prompts, train_completions], [
        test_prompts, test_completions], [
            valid_prompts, valid_completions]


def create_prompts(df: pd.DataFrame,
                   prompt_creator: Callable[[pd.Series],
                                            tuple[str,
                                                  str]]) -> list[list[str]]:
    assert prompt_creator is not None, "prompt_creator is None, please provide a template function to create your prompts"

    prompts, completions = [], []
    for _, row in df.iterrows():
        prompt_example, completion_example = prompt_creator(row)
        prompts.append(prompt_example)
        completions.append(completion_example)

    return [prompts, completions]


def create_source_domain_prompt(row: pd.Series) -> tuple[str, str]:
    prompt = "Extract the source domain for the following conceptual metaphor:\n"
    prompt += "Sentence: " + row['Example'] + "\n"
    prompt += "Target Domain: " + str(row['Target Domain']) + "\n"
    prompt += "Source Domain:"
    completion = str(row['Source Domain'])
    return prompt, completion


def create_source_domain_v2_prompt(row: pd.Series) -> tuple[str, str]:
    prompt = "Context: In linguistics, conceptual metaphors consists of understanding a given concept in terms of another\n"
    prompt += "Task: Extract the source domain from the sentence\n"
    prompt += f"Sentence: {row['example']}\n"
    prompt += f"Target domain: {row['target_domain']}\n"
    prompt += "Answer:"
    completion = str(row['source_domain'])
    return prompt, completion


def create_target_domain_v2_prompt(row: pd.Series) -> tuple[str, str]:
    prompt = "Context: In linguistics, conceptual metaphors consists of understanding a given concept in terms of another\n"
    prompt += "Task: Extract the target domain from the sentence\n"
    prompt += f"Sentence: {row['example']}\n"
    prompt += f"Source domain: {row['source_domain']}\n"
    prompt += "Answer:"
    completion = str(row['target_domain'])
    return prompt, completion


def create_source_lexeme_prompt(row: pd.Series) -> tuple[str, str]:
    prompt = "Context: In linguistics, conceptual metaphors consists of understanding a given concept in terms of another\n"
    prompt += "Task: Extract the source lexeme from the sentence\n"
    prompt += f"Sentence: {row['example']}\n"
    prompt += f"Target lexeme: {row['target_lexeme']}\n"
    prompt += "Answer:"
    completion = str(row['source_lexeme'])
    return prompt, completion


def create_target_lexeme_prompt(row: pd.Series) -> tuple[str, str]:
    prompt = "Context: In linguistics, conceptual metaphors consists of understanding a given concept in terms of another\n"
    prompt += "Task: Extract the target lexeme from the sentence\n"
    prompt += f"Sentence: {row['example']}\n"
    prompt += f"Source lexeme: {row['source_lexeme']}\n"
    prompt += "Answer:"
    completion = str(row['target_lexeme'])
    return prompt, completion


def get_binary_completion(label: str) -> str:
    if label == "0" or label == "literal":
        return "no"
    return "yes"


def create_metaphor_classification_prompt(row: pd.Series) -> tuple[str, str]:
    prompt = f"Sentence: {str(row['text']).strip()}\n"
    prompt += "Question: Is the sentence metaphoric?\n"
    prompt += "Answer:"
    completion = get_binary_completion(str(row['label']))
    return prompt, completion


def create_target_prompt(row: pd.Series) -> tuple[str, str]:
    prompt = "Extract the conceptual metaphor from the following sentence:\n"
    prompt += "Sentence: " + row['Example'] + "\n"
    prompt += "Source Domain: " + str(row['Source Domain']) + "\n"
    prompt += "Target Domain:"
    completion = str(row['Target Domain'])
    return prompt, completion


def create_full_prompt(row: pd.Series) -> tuple[str, str]:
    prompt = "Extract the conceptual metaphor from the following sentence:\n"
    prompt += "Sentence: " + row['Example'] + "\n"
    completion = "Target Domain: " + str(row['Target Domain']) + "\n"
    completion += "Source Domain: " + str(row['Source Domain'])
    return prompt, completion


def create_fewshot_prompts(
        prompts_train,
        completions_train,
        prompts_test,
        completions_test,
        train_indices,
        extra_def=False):
    prompts = []
    completions = []
    train_prompt = ""
    for i in train_indices:
        train_prompt += prompts_train[i] + " " + completions_train[i] + "\n"
    for i in range(len(prompts_test)):
        prompts.append(train_prompt + prompts_test[i])
        completions.append(completions_test[i])
    return prompts, completions


def prefix_prompts_with_few_shot_examples(
        train_examples: list[list[str]],
        test_examples: list[list[str]],
        few_shot_sample_indices: list[int]) -> list[str]:
    train_prompts, train_completions = train_examples[0], train_examples[1]
    test_prompts = test_examples[0]
    prompts = []
    train_prompt = ""
    for i in few_shot_sample_indices:
        train_prompt += train_prompts[i] + " " + train_completions[i] + "\n"
    for i in range(len(test_prompts)):
        prompts.append(train_prompt + test_prompts[i])
    return prompts


# additional prompt variant for step by step reasoning generations


def create_reasoning_prompts(prompts_test, completions_test):
    prompts = []
    completions = []
    train_prompt = '''Prompt: Extract the conceptual metaphor from the following sentence:
Sentence: He recovered his hopes for a peace on earth.
Target Domain: hope
Reasoning: In the sentence above, the hopes are recovered. Recovering has a basic physical sentence, that is, getting back an object that belonged to you previously. Thus, hopes are talked about as possessions or physical objects.
Source Domain: possessions

Prompt: Extract the conceptual metaphor from the following sentence:
Sentence:  He finally found the key to the problem.
Target Domain: problem
Reasoning: The problem has a key, thus is a container that can be opened.
Source Domain: container

Prompt: Extract the conceptual metaphor from the following sentence:
Sentence: You have to weigh the pros and cons.
Target Domain: comparison
Reasoning: In the sentence above, pros and cons are being compared. Instead of using the abstract word compare, the sentence uses the verb “to weight” usually used to describe measuring the weight of physical objects. Here, however, arguments, thus, non-physical entities, are being measures.
Source Domain: measuring weight

Prompt: Extract the conceptual metaphor from the following sentence:
Sentence: the contagion of democratic ideas
Target Domain: belief
Reasoning: In the sentence above, ideas are described as something that is contagious, like a disease.
Source Domain: disease

Prompt: Extract the conceptual metaphor from the following sentence:
Sentence: Follow your reasoning where it takes you.
Target Domain: logic
Reasoning: In the above sentence, logic refers to reasoning. Reasoning is described as something that you can follow, just like a path in the physical world.
Source Domain: path

Prompt: Extract the conceptual metaphor from the following sentence:
Sentence: But he said, don't wash it I wanna wear it.
Target Domain: washing
Reasoning: Washing is used in a literal sense and no metaphoric transfer is taking place.
Source Domain: not metaphoric\n\n'''
    for i in range(len(prompts_test)):
        # replace Source Domain with Reasoning in prompts_test[i]
        p_test = prompts_test[i].replace("Source Domain", "Reasoning")
        prompts.append(train_prompt + p_test)
        completions.append(completions_test[i])
    return prompts, completions
