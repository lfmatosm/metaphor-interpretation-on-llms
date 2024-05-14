import pandas as pd
import os


def main():
    div = [4, 6]
    for idx, dataset in enumerate(["vua_verb", "vua_pos"]):
        train_df = pd.read_csv(f"./Data/metaphors_in_plm/{dataset}/train.csv", index_col=0)
        test_df = pd.read_csv(f"./Data/metaphors_in_plm/{dataset}/test.csv", index_col=0)
        dev_df = pd.read_csv(f"./Data/metaphors_in_plm/{dataset}/dev.csv", index_col=0)
        print(len(train_df), len(test_df), len(dev_df))
        new_train_len = round(len(train_df) / div[idx])
        new_train_len += new_train_len % 2
        new_test_len = round(len(test_df) / div[idx])
        new_test_len += new_test_len % 2
        new_dev_len = round(len(dev_df) / div[idx])
        new_dev_len += new_dev_len % 2
        print(new_train_len, new_test_len, new_dev_len)

        new_train_df = train_df[train_df["label"] == 1].sample(n=int(new_train_len / 2), random_state=1)
        new_train_df = pd.concat([new_train_df, train_df[train_df["label"] == 0].sample(
            n=int(new_train_len / 2), random_state=1)]).sample(frac=1)
        new_test_df = test_df[test_df["label"] == 1].sample(n=int(new_test_len / 2), random_state=1)
        new_test_df = pd.concat([new_test_df, test_df[test_df["label"] == 0].sample(
            n=int(new_test_len / 2), random_state=1)]).sample(frac=1)
        new_dev_df = dev_df[dev_df["label"] == 1].sample(n=int(new_dev_len / 2), random_state=1)
        new_dev_df = pd.concat([new_dev_df, dev_df[dev_df["label"] == 0].sample(
            n=int(new_dev_len / 2), random_state=1)]).sample(frac=1)

        os.makedirs(f"./Data/metaphors_in_plm/{dataset}_subset/", exist_ok=True)
        new_train_df.to_csv(f"./Data/metaphors_in_plm/{dataset}_subset/train.csv")
        new_test_df.to_csv(f"./Data/metaphors_in_plm/{dataset}_subset/test.csv")
        new_dev_df.to_csv(f"./Data/metaphors_in_plm/{dataset}_subset/dev.csv")


if __name__ == "__main__":
    main()
