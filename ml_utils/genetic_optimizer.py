import numpy as np


class GeneticOptimizer:
    def __init__(self):
        pass

    def generation_log(self, generation_idx, best_fitness, best_fitnesses,
                       best_genome):
        pass

    @staticmethod
    def crossover(genomes):
        return np.array([g[np.random.random() > 0.5] for g in zip(*genomes)])

    @staticmethod
    def get_random_couples(size):
        if size == 1:
            couples = [(0, 0)]
        elif size == 2:
            r = np.random.random() > 0.5
            couples = [(int(r), int(not r))]
        else:
            couples = set()
            for i in range(size):
                j = i
                while j == i or (j, i) in couples:
                    j = np.random.randint(0, size)
                couples.add((i, j))
            couples = list(couples)
        return couples

    def evolve(
        self,
        pop_size=100,
        generations=60,
        best_threshold=0.3,
    ):
        best_count = int(np.floor(best_threshold * pop_size))
        genomes = np.array([self.get_random_genome() for _ in range(pop_size)])
        for gen_n in range(generations):
            print(f"\n*** Generation {gen_n + 1} ***\n")
            fitnesses = np.array(list(map(self.get_fitness, genomes)))
            best_idxs = np.argsort(fitnesses)[:best_count]
            best_genomes = genomes[best_idxs]
            random_couples = GeneticOptimizer.get_random_couples(
                len(best_genomes))
            print('Computing offspring')
            offspring = [
                self.mutate(
                    GeneticOptimizer.crossover(
                        (best_genomes[i], best_genomes[j])),
                    list(zip(best_genomes[i], best_genomes[j])))
                for i, j in random_couples
            ]
            self.generation_log(gen_n, fitnesses[best_idxs[0]],
                                fitnesses[best_idxs], genomes[best_idxs[0]])

            genomes = np.concatenate((best_genomes, offspring, [
                self.get_random_genome()
                for _ in range(pop_size - 2 * len(best_idxs))
            ]))

        best_idx = np.argmax(list(map(self.get_fitness, genomes)))
        return genomes[best_idx]


if __name__ == '__main__':
    print(GeneticOptimizer.get_random_couples(2))
