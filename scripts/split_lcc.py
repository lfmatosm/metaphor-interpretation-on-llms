import pandas as pd


def main():
    df = pd.read_csv("./scripts/cms.csv", sep=";", index_col=0)
    print("df len:", len(df))
    # combinations of source/target domain
    combinations = df.groupby(["source_domain", "target_domain"]).size().reset_index(name="counts")
    # obtains combinations with at least 20 counts
    most_freq_combinations = combinations[combinations["counts"] > 20].sort_values(by="counts", ascending=False)

    # filters df with selected combinations
    df_redux = pd.DataFrame()
    for _, row in most_freq_combinations.iterrows():
        df_redux = pd.concat([df_redux, df[(df["source_domain"] == row["source_domain"])
                             & (df["target_domain"] == row["target_domain"])]])

    subset_df = df_redux.copy(deep=True)
    print("redux df len:", len(subset_df))

    # train, test, dev splits
    # splits examples as evenly as possible considering the original domain combination distribution
    splits = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    split_fractions = [0.7, 0.2, 0.1]
    for i in range(len(split_fractions)):
        split_fraction = split_fractions[i]
        split_size = round(len(df_redux) * split_fraction)
        print(f"split_size: {split_size}\n")
        for _, row in most_freq_combinations.iterrows():
            combination_rows = subset_df[(subset_df["source_domain"] == row["source_domain"]) &
                                         (subset_df["target_domain"] == row["target_domain"])]
            # print(f"size: {split_size}, comb rows: {len(combination_rows)}, comb sample: {round(combination_fraction*split_size)}\n")
            frac_sample = round(row["counts"] * split_size / len(df_redux))
            # guaratees that no more samples than the necessary will be assigned to the split
            n_sample = frac_sample if frac_sample <= len(combination_rows) else len(combination_rows)
            n_sample = n_sample if n_sample <= split_size - len(splits[i]) else split_size - len(splits[i])
            combination_sample = combination_rows.sample(n=n_sample, random_state=1)
            subset_df = subset_df.drop(list(combination_sample.index))
            splits[i] = pd.concat([splits[i], combination_sample])

    # balance and scramble the splits
    for i in range(len(splits)):
        split_fraction = split_fractions[i]
        split_size = round(len(df_redux) * split_fraction)
        if len(splits[i]) < split_size:
            diff = split_size - len(splits[i])
            diff = diff if diff <= len(subset_df) else len(subset_df)
            print(f"split len: {len(splits[i])}, split size: {split_size}, diff: {diff}\n")
            sample = subset_df.sample(n=diff, random_state=1)
            subset_df = subset_df.drop(list(sample.index))
            splits[i] = pd.concat([splits[i], sample])
        splits[i] = splits[i].sample(frac=1, random_state=1).reset_index(drop=True)
        splits[i].to_csv("./scripts/split" + str(i) + ".csv", sep=";")


if __name__ == "__main__":
    main()
