from collections import Counter
import numpy as np


def words_of_len(file_name, n):
    with open(file_name, 'r') as word_list:
        return [word for word in word_list.read().splitlines() if len(word) == n]

def main():
    z = 5

    targets = words_of_len('wordlist.txt', z)
    words = words_of_len('validGuesses.txt', z)

    letters = np.array(sorted(set().union(*map(set, words))))
    letters_decode = {letter: ix for ix, letter in enumerate(letters)}

    words_ix = np.array([[letters_decode[letter] for letter in word] for word in words])
    targets_ix = np.array([[letters_decode[letter] for letter in word] for word in targets])

    n = len(words)
    m = len(targets)

    compares = words_ix[:, None, :, None] == targets_ix[None, :, None, :]

    sat_counts = np.empty((n, m), dtype=int)
    for i, guess_word in enumerate(words_ix):
        guess_str = words[i]
        for j, target_word in enumerate(targets_ix):
            target_str = targets[j]

            correct_letters = np.diagonal(compares[i, j])
            present_letters = np.any(compares[i, j], axis=-1) & ~correct_letters
            absent_letters = ~(correct_letters | present_letters)

            correct_sat = np.all(targets_ix[:, correct_letters] == guess_word[correct_letters], axis=1)
            absent_sat = np.all(targets_ix[:, :, None] != guess_word[absent_letters], axis=(1, 2))

            present_compare = targets_ix[:, :, None] == guess_word[present_letters]
            present_compare[:, np.arange(z)[present_letters], np.arange(np.sum(present_letters))] = False # cannot match present letters at their locations
            present_sat = np.all(np.any(present_compare, axis=1), axis=1)

            sat = correct_sat & present_sat & absent_sat
            sat_count = np.sum(sat)
            if sat_count == 0:
                sat_count = n
            sat_counts[i, j] = sat_count

        scores = np.mean(sat_counts[:i+1], axis=1)
        best = np.argmin(scores[np.isfinite(scores)])
        best = np.arange(i+1)[np.isfinite(scores)][best]
        print(f"Step {i+1}/{n} ({(i+1)/n:0.4f}%): '{words[i]}'={np.mean(sat_counts[i]):0.4f}, best: '{words[best]}'={np.mean(sat_counts[best]):0.4f}")

    # best = np.argmax(entropy)
    # print(f"Overall - best: '{words[best]}', entropy: {entropy[best]}")

if __name__ == '__main__':
    main()