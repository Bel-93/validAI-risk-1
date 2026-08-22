# Prueba unitaria del módulo de calibración (regla de negocio).
import numpy as np, pandas as pd
import calibracion as C


def _data(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    pd_est = rng.beta(2, 20, n)               # PD estimada baja
    default = (rng.random(n) < pd_est).astype(int)  # default coherente con la PD
    return pd.DataFrame({"pd": pd_est, "default": default})


def test_estructura_resultado():
    res = C.evaluar_calibracion(_data())
    assert "tabla_por_bucket" in res and not res["tabla_por_bucket"].empty
    assert "hosmer_lemeshow" in res and "spiegelhalter_z" in res
    assert "resumen_semaforo" in res


def test_modelo_bien_calibrado_predomina_verde():
    # Con default generado a partir de la PD, el modelo está calibrado:
    # el semáforo debe tener al menos un bucket VERDE.
    res = C.evaluar_calibracion(_data())
    assert res["resumen_semaforo"].get("VERDE", 0) >= 1
