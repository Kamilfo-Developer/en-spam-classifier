from pandas import read_csv
import re

dataset = read_csv("dataset/spambase_csv.csv")

WORDS = list(map(lambda x: x.split("word_freq_")[1], list(dataset)[0:48]))

CHARS = list(
    map(
        lambda x: chr(int(x.split("char_freq_")[1][1:], 16)),
        list(dataset)[48:54],
    )
)


def get_capital_run_length_total(
    string: str, regexp_pattern: str = r"[A-Z]+"
) -> int:
    all_capital_substrings = re.findall(regexp_pattern, string)

    return sum(map(len, all_capital_substrings))


def get_capital_run_length_average(
    string: str, regexp_pattern: str = r"[A-Z]+"
) -> float:
    all_capital_substrings = re.findall(regexp_pattern, string)

    return sum(map(len, all_capital_substrings)) / (
        len(all_capital_substrings) or 1
    )


def get_max_caital_run_from_string(
    string: str, regexp_pattern: str = r"[A-Z]+"
) -> int:
    longest_string = max(re.findall(regexp_pattern, string) or [""], key=len)

    return len(longest_string)


def count_substrings(string: str, substrings: list[str]) -> list[int]:
    result = list(map(lambda x: string.count(x), substrings))

    return result
