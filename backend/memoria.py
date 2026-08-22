# ============================================================
# ValidAI Risk — Backend: memoria de hallazgos (SQLite, MVP)
# Persistencia de auditoría desacoplada. En producción migra a
# PostgreSQL/RDS o DynamoDB por configuración (roadmap).
# En Cloud Run el filesystem es efímero: la memoria dura la vida
# del contenedor (decisión consciente de MVP; ver FAQ de defensa).
# ============================================================
import os
import sqlite3

import pandas as pd

DB_NAME = os.getenv("MEMORIA_DB", "memoria_validairisk.db")
_TABLAS_PERMITIDAS = ("hallazgos", "ejecuciones")


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hallazgos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modelo TEXT, periodo TEXT, categoria TEXT, hallazgo TEXT,
            severidad TEXT, impacto TEXT, recomendacion TEXT,
            decision_humana TEXT, comentario TEXT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ejecuciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modelo TEXT, periodo TEXT, estado TEXT, resumen TEXT, error TEXT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    conn.commit()
    conn.close()


def guardar_hallazgo_validado(modelo, periodo, categoria, hallazgo, severidad,
                              impacto, recomendacion, decision_humana, comentario):
    init_db()
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        """INSERT INTO hallazgos
           (modelo, periodo, categoria, hallazgo, severidad, impacto,
            recomendacion, decision_humana, comentario)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (modelo, periodo, categoria, hallazgo, severidad, impacto,
         recomendacion, decision_humana, comentario),
    )
    conn.commit()
    conn.close()


def guardar_ejecucion(modelo, periodo, estado, resumen, error=""):
    init_db()
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        "INSERT INTO ejecuciones (modelo, periodo, estado, resumen, error) VALUES (?,?,?,?,?)",
        (modelo, periodo, estado, resumen, error),
    )
    conn.commit()
    conn.close()


def consultar_hallazgos_previos(keyword: str, limite: int = 5) -> str:
    init_db()
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(
        "SELECT modelo, periodo, categoria, hallazgo, severidad, fecha "
        "FROM hallazgos WHERE hallazgo LIKE ? ORDER BY fecha DESC LIMIT ?",
        conn, params=(f"%{keyword}%", limite),
    )
    conn.close()
    if df.empty:
        return "No se encontraron hallazgos históricos."
    return df.to_string(index=False)


def obtener_tabla(tabla: str):
    if tabla not in _TABLAS_PERMITIDAS:
        return pd.DataFrame()
    init_db()
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(f"SELECT * FROM {tabla} ORDER BY fecha DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def conteo(tabla: str) -> int:
    if tabla not in _TABLAS_PERMITIDAS:
        return 0
    init_db()
    try:
        conn = sqlite3.connect(DB_NAME)
        n = conn.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return 0
