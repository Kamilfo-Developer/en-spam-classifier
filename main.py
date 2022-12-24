from text_preprocessing import (
    get_capital_run_length_average,
    get_capital_run_length_total,
    get_max_caital_run_from_string,
    count_substrings,
    WORDS,
    CHARS,
)
from train import model
from pandas import DataFrame, read_csv


text = input()

columns = list(read_csv("dataset/spambase_csv.csv").drop(["class"], axis=1))

df = DataFrame(
    [
        count_substrings(text.lower(), WORDS)
        + count_substrings(text.lower(), CHARS)
        + [get_capital_run_length_average(text)]
        + [get_max_caital_run_from_string(text)]
        + [get_capital_run_length_total(text)]
    ],
    columns=columns,
)


print(model.is_spam(df))
