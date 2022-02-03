import numpy as np

import dask
import dask.array as da
from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler, visualize


def words_of_len(file_name, n):
    with open(file_name, 'r') as word_list:
        return [word for word in word_list.read().splitlines() if len(word) == n]

def main():
    z = 5 # word len

    targets_str = words_of_len('wordlist.txt', z)
    guesses_str = words_of_len('validGuesses.txt', z) + targets_str

    # support non-alpha inputs
    letters = np.array(sorted(set().union(*map(set, guesses_str))))
    letters_decode = {letter: ix for ix, letter in enumerate(letters)}

    guesses = np.array([[letters_decode[letter] for letter in word] for word in guesses_str], dtype=np.int8)
    targets = np.array([[letters_decode[letter] for letter in word] for word in targets_str], dtype=np.int8)

    n = len(guesses)
    m = len(targets)

    cn = min(n, 128) # guesses chunk size n
    cm1 = min(m, 256) # targets chunk size m1
    cm2 = min(m, 2048) # targets chunk size m2

    compares = guesses[:, None, :, None] == targets[None, :, None, :] # bool:nmzz
    correct = np.diagonal(compares, axis1=-1, axis2=-2) # bool:nmz
    present = np.any(compares, axis=-1) & ~correct # bool:nmz
    absent = ~(correct | present) # bool:nmz

    correct_c = da.from_array(correct, chunks=(cn, cm1, z)) # bool:nmz
    present_c = da.from_array(present, chunks=(cn, cm1, z)) # bool:nmz
    absent_c = da.from_array(absent, chunks=(cn, cm1, z)) # bool:nmz
    targets_c = da.from_array(targets, chunks=(cm2, z)) # int:mz

    #correct_sat
    null_ix = -1 # sentinal index value for matching itself and not other indexes
    correct_guesses = np.where(correct_c[:, :, None, :], guesses[:, None, None, :], null_ix) # int:nm1z
    correct_targets = da.where(correct_c[:, :, None, :], targets_c[None, None, :, :], null_ix) # int:nmmz
    correct_compares = correct_guesses == correct_targets # bool:nmmz
    correct_sat = da.all(correct_compares, axis=-1) # bool:nmm

    # present_sat
    present_guesses = np.where(present_c[:, :, None, :], guesses[:, None, None, :], null_ix)  # int:nm1z
    present_compares = present_guesses[..., None] == targets_c[None, None, :, None, :] # bool:nmmzz
    present_compares = da.where(~present[:, :, None, :, None], present_compares, False) # bool:nmmzz - letter must be found elsewhere
    present_sat = da.any(present_compares, axis=-1) # bool:nmmz
    present_sat = present_sat | ~present[:, :, None, :] # bool:nmmz - ignore other results types
    present_sat = da.all(present_sat, axis=-1) # bool:nmm

    #absent_sat
    absent_guesses = da.where(absent_c[:, :, None, :], guesses[:, None, None, :], null_ix) # int:nm1z
    absent_compares = absent_guesses[:, :, :, :, None] != targets_c[None, None, :, None, :] # bool:nmmzz
    absent_sat = da.all(absent_compares, axis=(-1, -2))

    #compute top
    sat = correct_sat & present_sat & absent_sat # bool:nmm
    sat_avg = sat.sum(axis=-1).mean(axis=-1) # float:n
    best_ix = da.argtopk(-sat_avg, k=10) # int:k
    worst_ix = da.argtopk(sat_avg, k=10) # int:k

    pbar = ProgressBar()
    prof = Profiler()
    rprof = ResourceProfiler()
    cprof = CacheProfiler()
    with pbar, prof, rprof, cprof:
        best_ix, worst_ix = dask.compute(best_ix, worst_ix)

    best_guesses = np.take(guesses_str, best_ix)
    worst_guesses = np.take(guesses_str, worst_ix)
    print(f'best:{best_guesses}')
    print(f'worst:{worst_guesses}')

    visualize([prof, rprof, cprof], filename='profile.html')


if __name__ == '__main__':
    main()