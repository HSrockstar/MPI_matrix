import importlib.util
import sys
from pathlib import Path
import random
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab"))
spec = importlib.util.spec_from_file_location(
    "exact", Path(__file__).parents[1] / "lab/exact.py"
)
exact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exact)


class ExactModels(unittest.TestCase):
    def test_models(self):
        rng = random.Random(1234567)
        for m, k, n in [(1, 1, 1), (3, 7, 5), (4, 4, 4), (6, 11, 9), (2, 17, 3)]:
            for value in (None, 99, -99):
                a = [
                    [
                        rng.randrange(-99, 100) if value is None else value
                        for _ in range(k)
                    ]
                    for _ in range(m)
                ]
                b = [
                    [rng.randrange(-99, 100) if value is None else 99 for _ in range(n)]
                    for _ in range(k)
                ]
                ref = exact.naive(a, b)
                for name, fn in exact.MODELS.items():
                    with self.subTest(name=name, shape=(m, k, n), value=value):
                        self.assertEqual(fn(a, b), ref)

    def test_rectangular_certificate(self):
        import json

        scheme = json.loads(
            (Path(__file__).parents[1] / "schemes/strassen_2x2x3.json").read_text()
        )
        exact.certify(scheme)
        a = [[i * 13 + j - 70 for j in range(7)] for i in range(5)]
        b = [[i * 11 - j + 21 for j in range(8)] for i in range(7)]
        self.assertEqual(exact.tensor(a, b, scheme), exact.naive(a, b))
        self.assertEqual(exact.tensor(a, b, scheme, basis=True), exact.naive(a, b))

    def test_ntt_guard(self):
        with self.assertRaises(ValueError):
            exact.ntt_gemm([[10**100]], [[10**100]])

    def test_invalid_certificate(self):
        scheme = exact.strassen_scheme()
        scheme["U"][0][0] = 2
        with self.assertRaises(ValueError):
            exact.certify(scheme)

    def test_large_integers(self):
        a = [[10**60, -(10**40)], [10**30, 123]]
        b = [[10**20, -789], [10**40, 10**30]]
        for fn in (exact.kronecker, exact.crt, exact.structured):
            self.assertEqual(fn(a, b), exact.naive(a, b))

    def test_signed_bit_boundary(self):
        a = [[-128, 127, 0, -1]]
        b = [[-128], [127], [127], [-128]]
        for fn in (exact.bitserial, exact.four_russians):
            self.assertEqual(fn(a, b), exact.naive(a, b))


if __name__ == "__main__":
    unittest.main()
