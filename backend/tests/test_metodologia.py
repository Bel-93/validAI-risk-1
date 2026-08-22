# Prueba unitaria del módulo de validación metodológica (regla de negocio).
import numpy as np, pandas as pd
import validacion_metodologica as vm


def _data():
    np.random.seed(7); n = 3000
    deuda = np.random.rand(n); antig = np.random.rand(n); ing = np.random.randn(n)
    p = 1 / (1 + np.exp(-(-2.5 + 2.2 * deuda - 1.5 * antig + 0.3 * np.random.randn(n))))
    y = np.random.binomial(1, p)
    return pd.DataFrame({"deuda_ratio": deuda, "antiguedad": antig, "ingreso_z": ing,
                         "default": y, "pd": p, "pd_replicada": p + np.random.randn(n) * 0.01,
                         "periodo": np.random.choice(["202401", "202412"], n)})


def test_iv_variable_fuerte():
    r = vm.validar_metodologia(_data(),
        especificacion={"coeficientes": {"deuda_ratio": 2.2, "antiguedad": -1.5, "ingreso_z": 0.05}})
    assert r["variables"]["poder_predictivo"]["deuda_ratio"]["IV"] > 0.3


def test_gini_minimo_y_replicacion():
    r = vm.validar_metodologia(_data(),
        especificacion={"coeficientes": {"deuda_ratio": 2.2, "antiguedad": -1.5}})
    assert r["benchmark"]["modelo"]["Gini"] > 0.3
    assert r["benchmark"]["replicacion"]["estado"] == "VERDE"
