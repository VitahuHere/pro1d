import csv
import pprint


def bayes_probability(data, decision_attr, index, value):
    count = 0
    for d in data:
        if d[-1] == decision_attr and d[index] == value:
            count += 1
    return count / sum(d[-1] == decision_attr for d in data)


def build_probability_tree(
    data: list[list[str]], decision_attrs: set[str]
) -> dict[str, dict[int, dict[str, float]]]:
    probability_tree = {}
    for decision_attr in decision_attrs:
        probability_tree[decision_attr] = {}
        for i in range(0, len(data[0]) - 1):
            probability_tree[decision_attr][i] = {}
            for value in set(d[i] for d in data):
                probability_tree[decision_attr][i][value] = bayes_probability(
                    data, decision_attr, i, value
                )
    return probability_tree


def classify_bayes(
    data: list[list[str]],
    test_vector: list,
    decision_attrs: set[str],
    probability_tree: dict[str, dict[int, dict[str, float]]],
) -> str:
    probabilities = {}
    for decision_attr in decision_attrs:
        probabilities[decision_attr] = 1
        for i, value in enumerate(test_vector):
            probabilities[decision_attr] *= probability_tree[decision_attr][i][value]
        probabilities[decision_attr] *= sum(d[-1] == decision_attr for d in data) / len(
            data
        )
    return max(probabilities, key=probabilities.get)


def main():
    train = list(csv.reader(open("cars_evaluation.trn", "r")))
    test = list(csv.reader(open("cars_evaluation.tst", "r")))
    data = train + test
    decision_attrs = set(d[-1] for d in data)
    tree = build_probability_tree(train, decision_attrs)
    pprint.pp(tree)
    accurate = 0
    for t in test:
        pred = classify_bayes(train, t[:-1], decision_attrs, tree)
        if pred == t[-1]:
            accurate += 1

    print(accurate / len(test) * 100)


if __name__ == "__main__":
    main()
