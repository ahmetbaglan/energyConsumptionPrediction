import unittest


class TestSum(unittest.TestCase):
    def test_length(self,X,y):
        """
        Assert if input matrix and training y has same lenght
        :param X:
        :param y:
        :return:
        """
        self.assertEqual(len(X), len(y))

    def test_positive_pred(self,y):
        """
        Assert if all the forecasts are positive
        :param y:
        :return:
        """
        self.assertTrue((y>0).all())

    def test_max(self, y):
        """
        Tests if maximum forecast if less than 25 times that ever happened
        :param y:
        :return:
        """
        max_ever_possible = 4.72 * 25
        self.assertTrue(y.max()< max_ever_possible)

if __name__ == '__main__':
    unittest.main()
