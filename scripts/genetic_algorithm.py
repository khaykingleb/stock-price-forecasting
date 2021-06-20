from dataclasses import dataclass
import gc

import numpy as np

import torch
from torch import nn
from torch import optim


@dataclass
class GeneticAlgorithmConfig:
    ell: int = 20
    k: int = 5
    mutation_rate: float = 0.1
    num_epochs: int = 10
      
 
class Individual:
    def __init__(self):
        self.name = '#' + ''.join(map(str, np.random.randint(0,9, size=7).tolist()))
        self.num_epochs_base = np.random.choice(np.arange(60, 300))
        self.hidden_size = np.random.choice([2 ** power for power in range(2, 10)])
        self.num_layers = np.random.choice(np.arange(2, 15))
        self.learning_rate = round(np.random.random(), 2)

        self.loss = np.inf
        self.fitness = None

    def __repr__(self):
        """
        For convenience only.
        """
        string = 'Chromosome ' + self.name + f' with the loss of {self.loss:.4}' + f' and {self.num_epochs_base} epochs:\n'
        string = string + f'learning_rate = {self.learning_rate:.4}, ' 
        string = string + f'num_layers = {self.num_layers}, '+  f'hidden_size = {self.hidden_size}'
        return string


@dataclass
class Population:
    def __init__(self, config: GeneticAlgorithmConfig):
        self.individuals = [Individual() for _ in range(config.ell)]
        self.top_k_individuals = None
        self.best_indivdual = None
        
        
class GeneticAlgorithm:
    def __init__(self, optimized_block, criterion, 
                 population: Population, config: GeneticAlgorithmConfig, 
                 device, verbose=True, seed: int = 77):
        self.optimized_block = optimized_block
        self.criterion = criterion
        self.population = population
        self.config = config
        self.device = device
        self.verbose = verbose
        self.seed = seed

        self.val_loss_history = []

    def fit(self, X_val, y_val):
        for epoch in range(self.config.num_epochs):
            self.evaluate(X_val, y_val, self.population)
            self.select(self.population)
            self.val_loss_history.append(self.population.best_indivdual.loss)

            offsprings = []
            for weak_individual in self.population.individuals[self.config.k:]:
                strong_individual = np.random.choice(self.population.top_k_individuals)
                offsprings.append(self.crossover(strong_individual, weak_individual))

            new_population = self.population.top_k_individuals + offsprings

            mutated_population = []
            for individual in new_population[1:]:
                mutated_population.append(self.mutate(individual))

            self.population.individuals = [self.population.best_indivdual] + mutated_population

            if self.verbose:
                clear_output(wait=True)
                print(f"Epoch: {epoch + 1}")
                
                plot_metric(self.criterion.__class__.__name__, 
                            val_metric=self.val_loss_history)
                
                print(f'{self.population.best_indivdual}')
    
    def evaluate(self, X_val, y_val, population: Population):
        losses = []

        for individual in population.individuals:
            gc.collect()
            torch.cuda.empty_cache()

            if self.optimized_block == 'LSTM':
                seed_everything(self.seed)
                model = LSTM(input_size=X_val.shape[2],
                             hidden_size=int(individual.hidden_size),
                             num_layers=individual.num_layers).to(self.device)
      
            elif self.optimized_block == 'GRU':
                seed_everything(self.seed)
                model = GRU(input_size=X_val.shape[2],
                            hidden_size=int(individual.hidden_size),
                            num_layers=individual.num_layers).to(self.device)

            else:
                raise ValueError('Only LSTM and GRU blocks are available for optimization.')

            optimizer = optim.Adam(model.parameters(), lr=individual.learning_rate)

            seed_everything(self.seed)

            train(model, self.criterion, optimizer, X_val, y_val, individual.num_epochs_base, 
                  verbose=False, return_loss_history=False, compute_test_loss=False)
          
            individual.loss = predict(model, X_val, y_val, self.criterion)

            losses.append(individual.loss)

            del model 
        
        for individual in population.individuals:
            individual.fitness = self.normalize(individual.loss, min(losses), max(losses))

    def normalize(self, z, loss_best, loss_worst) -> float:
        return (z - loss_worst) / (loss_best - loss_worst)

    def select(self, population: Population):
        ranked_population = sorted(population.individuals,
                                   key=lambda individual: individual.fitness,
                                   reverse=True)
        
        population.best_indivdual = ranked_population[0]

        population.top_k_individuals = ranked_population[:self.config.k]

    def crossover(self, strong_parent: Individual, weak_parent: Individual) -> Individual:
        offspring = Individual()
        
        prob = strong_parent.fitness / (strong_parent.fitness + weak_parent.fitness)
        
        if np.random.random() > prob:
            offspring.hidden_size = weak_parent.hidden_size 
        else:
            offspring.hidden_size = strong_parent.hidden_size

        if np.random.random() > prob:
            offspring.num_layers = weak_parent.num_layers
        else:
            offspring.num_layers = strong_parent.num_layers

        if np.random.random() > prob:
            offspring.learning_rate = weak_parent.learning_rate
        else:
            offspring.learning_rate = strong_parent.learning_rate
        
        if np.random.random() > prob:
            offspring.num_epochs_base = weak_parent.num_epochs_base
        else:
            offspring.num_epochs_base = strong_parent.num_epochs_base

        return offspring

    def mutate(self, individual: Individual) -> Individual:
        if np.random.random() < config.mutation_rate:
            individual.hidden_size = 2 ** (np.log2(individual.hidden_size) + np.random.randint(-1, 2))
            individual.hidden_size = int(np.array(individual.hidden_size).clip(2 ** 3, 2 ** 9))

        if np.random.random() < config.mutation_rate:
            individual.num_layers += np.random.randint(-2, 3)
            individual.num_layers = np.array(individual.num_layers).clip(2, 14)

        if np.random.random() < config.mutation_rate:
            individual.learning_rate += round(np.random.uniform(-0.1, 0.1), 2)
            individual.learning_rate = np.array(individual.learning_rate).clip(0.001, 1)

        if np.random.random() < config.mutation_rate:
            individual.num_epochs_base += np.random.randint(-30, 30)
            individual.num_epochs_base = np.array(individual.num_epochs_base).clip(10, 300)

        return individual
