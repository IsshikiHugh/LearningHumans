import time
import random
import joblib

def do_work(i):
    # sleep for random time, but all shorter than 1s.
    random.seed(i)
    sleep_time = random.random()
    print(f'{i}: sleep {sleep_time:.2f}s...')
    time.sleep(sleep_time)
    print(f'{i}: awake!')
    return i

def main():
    results = joblib.Parallel(n_jobs=5)(joblib.delayed(do_work)(i) for i in range(10))
    print(f'Gathered results: {results}')