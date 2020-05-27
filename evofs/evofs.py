# -*- coding: utf-8 -*-
#
# Copyright 2019 Pietro Barbiero, Giovanni Squillero and Alberto Tonda
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import random
import copy
import traceback

import inspyred
import datetime
import numpy as np
import multiprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


class EvoFS(BaseEstimator, TransformerMixin):
    """
    EvoFS class.
    """

    def __init__(self, estimator, pop_size: int = 100, max_generations: int = 100, max_features: int = 100,
                 min_features: int = 10, n_splits: int = 3, random_state: int = 42,
                 scoring: str = "f1_weighted", verbose: bool = True):

        self.estimator = estimator
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.max_features = max_features
        self.min_features = min_features
        self.n_splits = n_splits
        self.random_state = random_state
        self.scoring = scoring
        self.verbose = verbose

    def fit(self, X, y=None, **fit_params):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        k = int(X.shape[1])

        self.max_generations_ = np.min([self.max_generations, int(math.log10(2**int(0.5 * k)))])
        self.pop_size_ = np.min([self.pop_size, int(math.log10(2**k))])
        self.offspring_size_ = 2 * self.pop_size_
        self.maximize_ = True
        self.individuals_ = []
        self.scorer_ = get_scorer(self.scoring)
        self.max_features_ = np.min([k, self.max_features])

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
        list_of_splits = [split for split in skf.split(X, y)]
        train_index, val_index = list_of_splits[0]

        self.x_train_, x_val = X.iloc[train_index], X.iloc[val_index]
        self.y_train_, y_val = y[train_index], y[val_index]

        # initialize pseudo-random number generation
        prng = random.Random()
        prng.seed(self.random_state)

        ea = inspyred.ec.emo.NSGA2(prng)
        ea.variator = [self._variate]
        ea.terminator = inspyred.ec.terminators.generation_termination
        ea.observer = self._observe

        ea.evolve(
            generator=self._generate,

            evaluator=self._evaluate,
            # this part is defined to use multi-process evaluations
            # evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
            # mp_evaluator=self._evaluate_feature_sets,
            # mp_num_cpus=multiprocessing.cpu_count()-2,

            pop_size=self.pop_size_,
            num_selected=self.offspring_size_,
            maximize=self.maximize_,
            max_generations=self.max_generations_,

            # extra arguments here
            current_time=datetime.datetime.now()
        )

        # find best individual, the one with the highest accuracy on the validation set
        accuracy_best = 0
        feature_counts = np.zeros(X.shape[1])
        for individual in ea.archive:

            feature_set = individual.candidate
            feature_counts[feature_set] += 1

            x_reduced = self.x_train_[feature_set]

            model = copy.deepcopy(self.estimator)
            model.fit(x_reduced, self.y_train_)

            # compute validation accuracy
            accuracy_val = self.scorer_(model, x_val[feature_set], y_val)

            if accuracy_best < accuracy_val:
                self.best_set_ = feature_set
                accuracy_best = accuracy_val

        self.feature_ranking_ = np.argsort(feature_counts)
        return self

    def transform(self, X, **fit_params):
        if isinstance(X, pd.DataFrame):
            return X[self.best_set_]
        return X[:, self.best_set_]

    # initial random generation of feature sets
    def _generate(self, random, args):
        n_features = random.randint(self.min_features, self.max_features_)
        individual = np.random.choice(self.x_train_.shape[1], size=(n_features,), replace=False).tolist()
        individual = np.sort(individual).tolist()
        return individual

    # using inspyred's notation, here is a single operator that performs both crossover and mutation, sequentially
    def _variate(self, random, candidates, args):
        split_idx = int(len(candidates) / 2)
        fathers = candidates[:split_idx]
        mothers = candidates[split_idx:]
        next_generation = []

        for parent1, parent2 in zip(fathers, mothers):

            # well, for starters we just crossover two individuals, then mutate
            children = [list(parent1), list(parent2)]

            # one-point crossover!
            cut_point1 = random.randint(1, len(children[0])-1)
            cut_point2 = random.randint(1, len(children[1])-1)
            child1 = children[0][cut_point1:] + children[1][:cut_point2]
            child2 = children[1][cut_point2:] + children[0][:cut_point1]

            # remove duplicates
            child1 = np.unique(child1).tolist()
            child2 = np.unique(child2).tolist()
            children = [child1, child2]

            # mutate!
            for child in children:
                mutation_point = random.randint(0, len(child)-1)
                while True:
                    new_val = np.random.choice(self.x_train_.shape[1])
                    if new_val not in child:
                        child[mutation_point] = new_val
                        break

            # check if individual is still valid, and
            # (in case it isn't) repair it
            for child in children:

                # if it has too many features, delete them
                if len(child) > self.max_features_:
                    n_surplus = len(child) - self.max_features_
                    indexes = np.random.choice(len(child), size=(n_surplus,))
                    child = np.delete(child, indexes).tolist()

                # if it has too less features, add more
                if len(child) < self.min_features:
                    n_surplus = self.min_features - len(child)
                    for _ in range(n_surplus):
                        while True:
                            new_val = np.random.choice(self.x_train_.shape[1])
                            if new_val not in child:
                                child.append(new_val)
                                break

            children[0] = np.sort(children[0]).tolist()
            children[1] = np.sort(children[1]).tolist()
            next_generation.append(children[0])
            next_generation.append(children[1])

        return next_generation

    # function that evaluates the feature sets
    def _evaluate(self, candidates, args):
        fitness = []
        for c in candidates:
            x_reduced = self.x_train_[c]
            model = copy.deepcopy(self.estimator)
            scores = cross_validate(model, x_reduced, self.y_train_, scoring=self.scorer_, cv=self.n_splits)
            cv_scores = np.mean(scores["test_score"])

            # compute numer of unused features
            features_removed = self.x_train_.shape[1] - len(c)

            # maximizing the points removed also means
            # minimizing the number of points taken (LOL)
            fitness.append(inspyred.ec.emo.Pareto([
                    features_removed,
                    cv_scores,
            ]))

        return fitness

    # the 'observer' function is called by inspyred algorithms at the end of every generation
    def _observe(self, population, num_generations, num_evaluations, args):

        feature_size = self.x_train_.shape[1]
        old_time = args["current_time"]
        # logger = args["logger"]
        current_time = datetime.datetime.now()
        delta_time = current_time - old_time

        # I don't like the 'timedelta' string format,
        # so here is some fancy formatting
        delta_time_string = str(delta_time)[:-7] + "s"

        log = "[%s] Generation %d, Random individual: size=%d, score=%.2f" \
            % (delta_time_string, num_generations,
               feature_size - population[0].fitness[0],
               population[0].fitness[1])
        if self.verbose:
            print(log)
        #     logger.info(log)

        args["current_time"] = current_time
