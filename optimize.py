from .__main__ import VAEManager
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import json
from torch.utils.tensorboard import SummaryWriter as TBSummaryWriter

from .ml_utils.genetic_optimizer import GeneticOptimizer
from .ml_utils.misc import CurrentDir, get_settings
from .ml_utils.summary_writer import SummaryWriter

curdir = CurrentDir(__file__)
settings = get_settings(curdir)


def rnd(a, b):
    return np.random.random() * (b - a) + a


class VAEGeneticOptimizer(GeneticOptimizer):
    def __init__(self):
        self.writer = SummaryWriter(curdir('genetic_optimization'))
        self.tb_writer = TBSummaryWriter(curdir('genetic_optimization'))

    @staticmethod
    def genome_to_dict(genome):
        return {
            'batch_size': int(genome[0]),
            'lr': float(genome[1]),
            'gamma': float(genome[2]),
            'weight_decay': float(genome[3])
        }

    def get_random_genome(self):
        return np.array([
            np.random.random() * (128 - 10) + 10,
            np.random.random(),
            np.random.random(),
            rnd(1e-7, 1e-4)
        ])

    def get_fitness(self, genome):
        local_settings = settings.copy()
        local_settings.update({
            'epochs': 1,
            'batch_size': int(genome[0]),
            'lr': genome[1],
            'gamma': genome[2],
            'weight_decay': genome[3]
        })
        VAE_manager = VAEManager(local_settings)
        VAE_manager.train()
        return VAE_manager.test()

    def mutate(self, offspring_genome, parent_genomes):

        for i, parent_genes in enumerate(parent_genomes):
            if np.random.random() < 0.2:
                mean = np.mean(parent_genes)
                var = np.abs(mean - parent_genes[0])
                offspring_genome[i] = np.random.randn() * var + mean
        return np.array([
            np.clip(offspring_genome[0], 10, 128),
            np.clip(offspring_genome[1], 1e-8, 1.0),
            np.clip(offspring_genome[2], 0, 1.0),
            np.clip(offspring_genome[3], 1e-7, 1e-4)
        ])

    def generation_log(self, generation_idx, best_fitness, best_fitnesses,
                       best_genome):
        self.writer.add_scalar('best_fitness',
                               float(best_fitness),
                               global_step=generation_idx)
        self.tb_writer.add_histogram('fitnesses',
                                     best_fitnesses,
                                     global_step=generation_idx)

        with open(curdir('evolved_params.json'), 'w') as f:
            json.dump(VAEGeneticOptimizer.genome_to_dict(best_genome), f)


def main():
    optimizer = VAEGeneticOptimizer()
    params = optimizer.evolve(pop_size=30)
    print(f"Evolved params: {params}")


if __name__ == "__main__":
    main()
