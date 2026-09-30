from symbolic import Symbolic


target = 10

if __name__ == "__main__":
    data = open("../../../data/irc-data/train/2004-12-25.train-c.ascii.txt").read()
    data = data.split("\n")

    evaluators = [Symbolic(name) for name in set(filter(lambda x: x is not None, map(Symbolic.extract_username, data)))]

    print(evaluators)

    message_scores = {}

    window = []

    for line in data:
        if len(window) == target:
            window.pop(0)

        window.append(line)

        # Skip windows ending on a system line ("=== x has joined") -- there's no speaker to score
        if len(window) == target and Symbolic.extract_username(window[-1]) is not None:
            scores = [(evaluator.get_name(), evaluator.score_text(window)) for evaluator in evaluators]
            message_scores[window[-1]] = scores

            print(scores)

    print("\n\n")

    winners = []

    for message in message_scores:
        winners.append((message, max(message_scores[message], key=lambda x: x[1])))

    print(winners)
