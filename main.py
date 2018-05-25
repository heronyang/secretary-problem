#!/usr/bin/env python3
"""
This is a simple experiment on the secretary problem and it's expected to see
the optimal pivot value is around 37%.
"""
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Decider():
    """
    A decider makes a decision based on the look-then-leap rule where it only
    looks and returns negative decision before the pivot point; and returns
    positive decision after the pivot point and if it is seeing the best option
    so far.
    """

    def __init__(self, pivot=0.37):
        self.pivot = pivot

    def decide(self, question):
        """
        Returns the index of entry this decider picks.
        """
        # Enters the leap stage
        start_index = int(question.length * self.pivot)
        for i in range(start_index, question.length):
            curr_val = question.get(i)
            best_val = question.get_best_value_between(0, i)
            if curr_val > best_val:
                return i
        # Worst case, we missed the best one, then just return the last one
        return question.length - 1


class Question():
    """
    A question contains a quence of numbers minicing the relative score of each
    candidate, and it also offers multiple methods for accessing the content of
    this sequence.
    """

    def __init__(self, length=1000):
        self.seq = [i for i in range(length)]
        random.shuffle(self.seq)
        self.answer = self.seq.index(max(self.seq))

    def get(self, index):
        """
        Gets the relative score of the candidate at a given index.
        """
        return self.seq[index]

    def get_best_value_between(self, start_index, end_index):
        """
        Gets the best relative score of the candidate given an index range.
        """
        subseq = self.seq[start_index:end_index]
        return max(subseq) if subseq else -1

    @property
    def length(self):
        """
        Returns the number of the candidates.
        """
        return len(self.seq)


def main():
    """
    Main function.
    """

    step = 0.01
    samples = 1000

    result = pd.DataFrame(columns=["correctness"])

    for pivot in np.arange(0, 1, step):
        decider = Decider(pivot)
        counter = 0
        for _ in range(samples):
            question = Question()
            if question.answer == decider.decide(question):
                counter += 1
        result.set_value(pivot, "correctness", counter / samples)
        print(pivot, counter / samples)

    result.to_csv("result.csv")
    save_plot(result, "result.png")


def save_plot(dataframe, filename):
    """
    Plot a dataframe and save it to disk.
    """
    plt.clf()
    dataframe.plot()
    plt.savefig(filename)


if __name__ == "__main__":
    main()
