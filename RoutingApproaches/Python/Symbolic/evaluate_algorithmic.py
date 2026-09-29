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

        if len(window) == target:
            line = [(evaluator.get_name(), evaluator.score_text(window)) for evaluator in evaluators]
            message_scores[window[-1]] = line

            print(line)
    
    print("\n\n")

    winners = []

    for message in message_scores:
        winners.append((message, max(message_scores[message], key=lambda x: x[1])))
    
    print(winners)
