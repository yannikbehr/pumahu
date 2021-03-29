import inspect
import os
import unittest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


class NBTestCase(unittest.TestCase):

    def setUp(self):
        self.cwd = os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe())))
        self.nbdir = os.path.join(self.cwd, '..', 'src', 'pumahu', 'notebook')

    def test_estimation(self):
        nbfn = os.path.join(self.nbdir, 'estimation.ipynb')
        with open(nbfn) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': self.nbdir}})

    def test_synthetics(self):
        nbfn = os.path.join(self.nbdir, 'forward_model.ipynb')
        with open(nbfn) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': self.nbdir}})



if __name__ == '__main__':
    unittest.main()

