import random

def _get_random_year_combination(self, num_years, size):
    sample = set()
    while len(sample) < size:
        # Choose one random item from items
        elem = random.randint(0, num_years - 1)
        # Using a set elminates duplicates easily
        sample.add(elem)
    return tuple(sample)


def get_random_year_combinations(self, num_combinations, num_years, size):
    samples = set()
    while len(samples) < num_combinations:
        comb = self._get_random_year_combination(num_years, size)
        samples.add(comb)

    return tuple(samples)
