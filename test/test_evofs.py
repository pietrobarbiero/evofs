import unittest

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier


class TestEvoCore(unittest.TestCase):

    def test_class(self):

        from evofs import EvoFS


        X, y = load_digits(return_X_y=True)

        model = EvoFS(RidgeClassifier(random_state=42))
        model.fit(X, y)
        x_reduced = model.transform(X)

        score1 = RidgeClassifier(random_state=42).fit(x_reduced, y).score(X[:, model.best_set_], y)
        score2 = RandomForestClassifier(random_state=42).fit(x_reduced, y).score(X[:, model.best_set_], y)

        print(score1)
        print(score2)

        return


suite = unittest.TestLoader().loadTestsFromTestCase(TestEvoCore)
unittest.TextTestRunner(verbosity=2).run(suite)
