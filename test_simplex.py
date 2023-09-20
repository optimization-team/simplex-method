# launch from cmd with: pytest test_simplex.py
from Simplex import Simplex
from parser import parse_test


class TestSimplex:
    def test_simplex(self):
        f, m, b, a, x_check, opt_check = parse_test('tests/test4.txt')
        simplex = Simplex(f, m, b, a)
        # opt, x = simplex.optimize()
        opt, x = simplex.plug_optimize()

        for i in range(len(x)):
            assert round(x[i], a) == x_check[i]
        assert opt == opt_check






