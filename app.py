import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    BaggingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------
# Configuración básica
# ------------------------------------------------------------------
BASE_FILE = "251107 Base colegios pricing (1).xlsx"

st.set_page_config(
    page_title="Simulador de pensión y matrícula",
    layout="wide",
)

# ------------------------------------------------------------------
# 1. Cargar y preparar la base
# ------------------------------------------------------------------
@st.cache_data
def load_and_prepare_base():
    df = pd.read_excel(BASE_FILE)

    # Ingeniería de variables
    df["SEGMENTO_GEOGRAFICO"] = np.where(
        df["LOCALIDAD"].notna(),
        df["MUNI"].astype(str) + " - " + df["LOCALIDAD"].astype(str),
        df["MUNI"].astype(str),
    )

    orden_icfes = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1}
    df["PLANTEL_ICFES_ORD"] = df["PLANTEL ICFES 2024"].map(orden_icfes)

    umbral_nuevo = 5
    df["COLEGIO_NUEVO"] = np.where(df["AÑOS DE OPERACIÓN"] <= umbral_nuevo, 1, 0)
    df["COLEGIO_NUEVO"] = df["COLEGIO_NUEVO"].fillna(0)

    educativas_base = [
        "PROM. PUNTAJE GLOBAL",
        "PROM. PUNTAJE INGLÉS",
        "PROM. PUNTAJE MATEMATICAS",
        "PROM. PUNTAJE LECTURA CRITICA",
        "PROM. PUNTAJE SOCIALES CIUDADANAS",
        "PROM. PUNTAJE CIENCIAS NATURALES",
    ]

    df["PROM_PUNTAJE_PROMEDIO_AREAS"] = df[
        [
            "PROM. PUNTAJE INGLÉS",
            "PROM. PUNTAJE MATEMATICAS",
            "PROM. PUNTAJE LECTURA CRITICA",
            "PROM. PUNTAJE SOCIALES CIUDADANAS",
            "PROM. PUNTAJE CIENCIAS NATURALES",
        ]
    ].mean(axis=1)

    df["PROM_PUNTAJE_STEM"] = df[
        ["PROM. PUNTAJE MATEMATICAS", "PROM. PUNTAJE CIENCIAS NATURALES"]
    ].mean(axis=1)

    df["PROM_PUNTAJE_LETRAS"] = df[
        [
            "PROM. PUNTAJE LECTURA CRITICA",
            "PROM. PUNTAJE SOCIALES CIUDADANAS",
            "PROM. PUNTAJE INGLÉS",
        ]
    ].mean(axis=1)

    df["DESVIO_PUNTAJES_AREAS"] = df[
        [
            "PROM. PUNTAJE INGLÉS",
            "PROM. PUNTAJE MATEMATICAS",
            "PROM. PUNTAJE LECTURA CRITICA",
            "PROM. PUNTAJE SOCIALES CIUDADANAS",
            "PROM. PUNTAJE CIENCIAS NATURALES",
        ]
    ].std(axis=1)

    df["BRECHA_GLOBAL_AREAS"] = (
        df["PROM. PUNTAJE GLOBAL"] - df["PROM_PUNTAJE_PROMEDIO_AREAS"]
    )

    meta = {
        "orden_icfes": orden_icfes,
        "umbral_nuevo": umbral_nuevo,
        "educativas_base": educativas_base,
    }

    return df, meta


df_base, meta = load_and_prepare_base()
orden_icfes = meta["orden_icfes"]
umbral_nuevo = meta["umbral_nuevo"]
educativas_base = meta["educativas_base"]

# Detectar posibles columnas de nombre y DANE en la base
def detectar_nombre_y_dane(df):
    name_candidates = [
        "NOMBRE_COLEGIO",
        "NOMBRE COLEGIO",
        "NOMBRE_ESTABLECIMIENTO",
        "NOMBRE ESTABLECIMIENTO",
        "NOMBRE INSTITUCIÓN",
        "NOMBRE_INSTITUCIÓN",
        "COLEGIO",
        "NOMBRE_SEDE",
        "NOMBRE SEDE",
    ]
    dane_candidates = [
        "COD_DANE",
        "CODIGO_DANE",
        "CÓDIGO_DANE",
        "CODIGO DANE",
        "DANE",
        "DANE_SEDE",
        "DANE ESTABLECIMIENTO",
    ]
    name_col = next((c for c in name_candidates if c in df.columns), None)
    dane_col = next((c for c in dane_candidates if c in df.columns), None)
    return name_col, dane_col

NAME_COL, DANE_COL = detectar_nombre_y_dane(df_base)

# ------------------------------------------------------------------
# 2. Entrenar modelo de pensión (ensemble) con cache
# ------------------------------------------------------------------
@st.cache_resource
def train_pension_model(df):

    educativas = educativas_base + [
        "PROM_PUNTAJE_PROMEDIO_AREAS",
        "PROM_PUNTAJE_STEM",
        "PROM_PUNTAJE_LETRAS",
        "DESVIO_PUNTAJES_AREAS",
        "BRECHA_GLOBAL_AREAS",
    ]

    economicas = [
        "INSE",
        "NSE_ ESTABLECIMIENTO",
        "NSE_ESTUDIANTE",
    ]

    geograficas = [
        "DEPTO",
        "MUNI",
        "SEGMENTO_GEOGRAFICO",
        "TIPO DE MUNICIPIO",
    ]

    otras_categoricas = [
        "CALENDARIO",
        "ES_BILINGUE",
        "JORNADA",
    ]

    def crear_preprocesador_para_pension():
        numeric_features = educativas + economicas + [
            "PLANTEL_ICFES_ORD",
            "AÑOS DE OPERACIÓN",
            "COLEGIO_NUEVO",
            "ESTUDIANTES_TOTALES_2023",
        ]
        categorical_features = geograficas + otras_categoricas

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        feature_cols = numeric_features + categorical_features
        return pre, feature_cols

    def obtener_modelos_candidatos_pension():
        modelos = {
            "rf": lambda rs: RandomForestRegressor(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=rs,
            ),
            "gbrt": lambda rs: GradientBoostingRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                random_state=rs,
            ),
            "extratrees": lambda rs: ExtraTreesRegressor(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=rs,
            ),
            "bagging_dt": lambda rs: BaggingRegressor(
                DecisionTreeRegressor(
                    max_depth=10,
                    min_samples_leaf=4,
                    random_state=rs,
                ),
                n_estimators=300,
                bootstrap=True,
                n_jobs=-1,
                random_state=rs,
            ),
        }
        return modelos

    def entrenar_y_evaluar_ensemble_pension(
        df,
        usar_log=True,
        random_state=42,
    ):
        modelos_candidatos = obtener_modelos_candidatos_pension()
        nombres_modelos = list(modelos_candidatos.keys())

        pre0, feature_cols = crear_preprocesador_para_pension()
        X_all = df[feature_cols].copy()
        y_raw_all = df["PENSIÓN"].astype(float).values

        def fwd(v):
            return np.log1p(v) if usar_log else v

        def inv(v):
            return np.expm1(v) if usar_log else v

        groups = df["MUNI"]
        gkf = GroupKFold(n_splits=5)

        metricas = {
            nombre: {"r2": [], "mae": [], "rmse": []}
            for nombre in nombres_modelos + ["ensemble"]
        }

        for train_idx, val_idx in gkf.split(X_all, y_raw_all, groups):
            X_train = X_all.iloc[train_idx]
            X_val = X_all.iloc[val_idx]
            y_train_raw = y_raw_all[train_idx]
            y_val_raw = y_raw_all[val_idx]

            y_train_tf = fwd(y_train_raw)
            preds_modelos = {}

            for nombre in nombres_modelos:
                pre_fold, _ = crear_preprocesador_para_pension()
                modelo = modelos_candidatos[nombre](random_state)

                pipe = Pipeline(
                    steps=[
                        ("preprocess", pre_fold),
                        ("model", modelo),
                    ]
                )

                pipe.fit(X_train, y_train_tf)
                y_pred_val_tf = pipe.predict(X_val)
                y_pred_val_raw = inv(y_pred_val_tf)
                preds_modelos[nombre] = y_pred_val_raw

                r2 = r2_score(y_val_raw, y_pred_val_raw)
                mae = mean_absolute_error(y_val_raw, y_pred_val_raw)
                mse = mean_squared_error(y_val_raw, y_pred_val_raw)
                rmse = np.sqrt(mse)

                metricas[nombre]["r2"].append(r2)
                metricas[nombre]["mae"].append(mae)
                metricas[nombre]["rmse"].append(rmse)

            matriz_preds = np.column_stack([preds_modelos[n] for n in nombres_modelos])
            y_pred_ens_raw = matriz_preds.mean(axis=1)

            r2 = r2_score(y_val_raw, y_pred_ens_raw)
            mae = mean_absolute_error(y_val_raw, y_pred_ens_raw)
            mse = mean_squared_error(y_val_raw, y_pred_ens_raw)
            rmse = np.sqrt(mse)

            metricas["ensemble"]["r2"].append(r2)
            metricas["ensemble"]["mae"].append(mae)
            metricas["ensemble"]["rmse"].append(rmse)

        resumen = {}
        for nombre in nombres_modelos + ["ensemble"]:
            r2_mean = np.mean(metricas[nombre]["r2"])
            mae_mean = np.mean(metricas[nombre]["mae"])
            rmse_mean = np.mean(metricas[nombre]["rmse"])
            resumen[nombre] = {
                "r2_mean": r2_mean,
                "mae_mean": mae_mean,
                "rmse_mean": rmse_mean,
            }

        mejor_nombre = max(resumen.keys(), key=lambda n: resumen[n]["r2_mean"])
        if mejor_nombre == "ensemble":
            nombres_seleccionados = nombres_modelos
        else:
            nombres_seleccionados = [mejor_nombre]

        modelos_finales = {}
        y_all_tf = fwd(y_raw_all)
        for nombre in nombres_seleccionados:
            pre_final, _ = crear_preprocesador_para_pension()
            modelo_final = modelos_candidatos[nombre](random_state + 999)

            pipe_final = Pipeline(
                steps=[
                    ("preprocess", pre_final),
                    ("model", modelo_final),
                ]
            )
            pipe_final.fit(X_all, y_all_tf)
            modelos_finales[nombre] = pipe_final

        return modelos_finales, usar_log, feature_cols, resumen

    modelos_pension, usar_log_pension, feat_pension, resumen_pension = (
        entrenar_y_evaluar_ensemble_pension(df, usar_log=True, random_state=42)
    )

    # Sigma residual en log(PENSIÓN)
    X_all = df[feat_pension]
    y_log_real = np.log1p(df["PENSIÓN"].astype(float).values)

    preds_log = []
    for _, pipe in modelos_pension.items():
        y_hat_tf = pipe.predict(X_all)
        y_hat_log = y_hat_tf  # entrenamos en log
        preds_log.append(y_hat_log)

    y_hat_log_mean = np.column_stack(preds_log).mean(axis=1)
    resid = y_log_real - y_hat_log_mean
    sigma_resid_log = resid.std()

    return modelos_pension, usar_log_pension, feat_pension, sigma_resid_log, resumen_pension


modelos_pension, usar_log_pension, feat_pension, sigma_resid_log, resumen_pension = (
    train_pension_model(df_base)
)

# ------------------------------------------------------------------
# 3. Funciones auxiliares
# ------------------------------------------------------------------
def preparar_nueva_muestra(df_nueva: pd.DataFrame) -> pd.DataFrame:
    df_nueva = df_nueva.copy()

    df_nueva["SEGMENTO_GEOGRAFICO"] = np.where(
        df_nueva["LOCALIDAD"].notna(),
        df_nueva["MUNI"].astype(str) + " - " + df_nueva["LOCALIDAD"].astype(str),
        df_nueva["MUNI"].astype(str),
    )
    df_nueva["PLANTEL_ICFES_ORD"] = df_nueva["PLANTEL ICFES 2024"].map(orden_icfes)
    df_nueva["COLEGIO_NUEVO"] = np.where(
        df_nueva["AÑOS DE OPERACIÓN"] <= umbral_nuevo, 1, 0
    )
    df_nueva["COLEGIO_NUEVO"] = df_nueva["COLEGIO_NUEVO"].fillna(0)

    df_nueva["PROM_PUNTAJE_PROMEDIO_AREAS"] = df_nueva[
        [
            "PROM. PUNTAJE INGLÉS",
            "PROM. PUNTAJE MATEMATICAS",
            "PROM. PUNTAJE LECTURA CRITICA",
            "PROM. PUNTAJE SOCIALES CIUDADANAS",
            "PROM. PUNTAJE CIENCIAS NATURALES",
        ]
    ].mean(axis=1)

    df_nueva["PROM_PUNTAJE_STEM"] = df_nueva[
        ["PROM. PUNTAJE MATEMATICAS", "PROM. PUNTAJE CIENCIAS NATURALES"]
    ].mean(axis=1)

    df_nueva["PROM_PUNTAJE_LETRAS"] = df_nueva[
        [
            "PROM. PUNTAJE LECTURA CRITICA",
            "PROM. PUNTAJE SOCIALES CIUDADANAS",
            "PROM. PUNTAJE INGLÉS",
        ]
    ].mean(axis=1)

    df_nueva["DESVIO_PUNTAJES_AREAS"] = df_nueva[
        [
            "PROM. PUNTAJE INGLÉS",
            "PROM. PUNTAJE MATEMATICAS",
            "PROM. PUNTAJE LECTURA CRITICA",
            "PROM. PUNTAJE SOCIALES CIUDADANAS",
            "PROM. PUNTAJE CIENCIAS NATURALES",
        ]
    ].std(axis=1)

    df_nueva["BRECHA_GLOBAL_AREAS"] = (
        df_nueva["PROM. PUNTAJE GLOBAL"] - df_nueva["PROM_PUNTAJE_PROMEDIO_AREAS"]
    )

    return df_nueva


def predecir_pension_ensemble(df_nueva_proc: pd.DataFrame) -> np.ndarray:
    X_new = df_nueva_proc[feat_pension]
    preds = []
    for _, pipe in modelos_pension.items():
        y_hat_tf = pipe.predict(X_new)
        if usar_log_pension:
            y_hat_raw = np.expm1(y_hat_tf)
        else:
            y_hat_raw = y_hat_tf
        preds.append(y_hat_raw)
    matriz = np.column_stack(preds)
    return matriz.mean(axis=1)


def resumen_estudiantes_por_depto_y_pension(
    depto: str,
    muni: str,
    pension_pred: float,
    z: float = 1.96,
):
    log_center = np.log1p(pension_pred)
    low_log = log_center - z * sigma_resid_log
    high_log = log_center + z * sigma_resid_log
    low = np.expm1(low_log)
    high = np.expm1(high_log)

    mask = df_base["DEPTO"] == depto
    if muni:
        mask &= df_base["MUNI"] == muni
    mask &= df_base["PENSIÓN"].between(low, high)

    df_filtrado = df_base.loc[mask].copy()
    prom_est_dep = df_base.loc[
        df_base["DEPTO"] == depto, "ESTUDIANTES_TOTALES_2023"
    ].mean()
    prom_est_muni_banda = (
        df_filtrado["ESTUDIANTES_TOTALES_2023"].mean()
        if len(df_filtrado) > 0
        else np.nan
    )

    return {
        "pension_low": low,
        "pension_high": high,
        "n_colegios_muni_banda": len(df_filtrado),
        "prom_est_dep": prom_est_dep,
        "prom_est_muni_banda": prom_est_muni_banda,
        "df_filtrado": df_filtrado,
    }


def chart_hist_with_line(series, value, title, x_label):
    series = series.dropna()
    if len(series) == 0 or value is None or np.isnan(value):
        return None

    data = pd.DataFrame({x_label: series})
    hist = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_label}:Q", bin=alt.Bin(maxbins=30)),
            y="count()",
        )
    )

    rule = (
        alt.Chart(pd.DataFrame({x_label: [value]}))
        .mark_rule(color="red", size=2)
        .encode(x=f"{x_label}:Q")
    )

    return (hist + rule).properties(title=title, width=400, height=250)


# ------------------------------------------------------------------
# 4. Interfaz Streamlit
# ------------------------------------------------------------------
def main():
    st.title("Simulador de pensión y matrícula esperada")
    st.markdown(
        "Este visualizador usa un modelo de **pensión** y, a partir de la banda de pensión, "
        "estima cuántos estudiantes suelen tener colegios similares en el mismo municipio."
    )

    # Selección de DEPTO y MUNI
    deptos = sorted(df_base["DEPTO"].dropna().unique())
    depto_sel = st.selectbox("Departamento (DEPTO)", deptos)

    munis = sorted(df_base.loc[df_base["DEPTO"] == depto_sel, "MUNI"].dropna().unique())
    muni_sel = st.selectbox("Municipio (MUNI)", munis)

    # Localidad dependiente de DEPTO + MUNI
    localidades_muni = (
        df_base.loc[
            (df_base["DEPTO"] == depto_sel) & (df_base["MUNI"] == muni_sel),
            "LOCALIDAD",
        ]
        .dropna()
        .unique()
    )
    localidades_muni = sorted(localidades_muni)

    st.subheader("Características del colegio")

    tipos_muni = sorted(df_base["TIPO DE MUNICIPIO"].dropna().unique())
    calendarios = sorted(df_base["CALENDARIO"].dropna().unique())
    bilingues = sorted(df_base["ES_BILINGUE"].dropna().unique())
    jornadas = sorted(df_base["JORNADA"].dropna().unique())
    icfes_opts = ["A+", "A", "B", "C", "D"]

    col1, col2, col3 = st.columns(3)

    with col1:
        tipo_muni_sel = st.selectbox("Tipo de municipio", tipos_muni)
        calendario_sel = st.selectbox("Calendario", calendarios)
        bilingue_sel = st.selectbox("¿Es bilingüe?", bilingues)
        jornada_sel = st.selectbox("Jornada", jornadas)

    with col2:
        anos_oper = st.number_input(
            "Años de operación", min_value=0, max_value=50, value=5
        )
        icfes_sel = st.selectbox("Plantel ICFES 2024", icfes_opts)
        inse = st.number_input(
            "INSE", value=float(df_base["INSE"].mean(skipna=True))
        )
        nse_estab = st.number_input(
            "NSE establecimiento",
            value=float(df_base["NSE_ ESTABLECIMIENTO"].mean(skipna=True)),
        )

    with col3:
        nse_estud = st.number_input(
            "NSE estudiante",
            value=float(df_base["NSE_ESTUDIANTE"].mean(skipna=True)),
        )
        est_tot = st.number_input(
            "Estudiantes totales estimados",
            min_value=1.0,
            value=float(df_base["ESTUDIANTES_TOTALES_2023"].median(skipna=True)),
        )

        if len(localidades_muni) > 0:
            opciones_loc = ["Sin localidad / NA"] + list(localidades_muni)
            loc_sel = st.selectbox("Localidad", opciones_loc)
            localidad_value = (
                np.nan if loc_sel == "Sin localidad / NA" else loc_sel
            )
        else:
            st.text("No hay información de localidad para este municipio.")
            localidad_value = np.nan

    st.subheader("Resultados Saber 11 (promedios)")
    col4, col5, col6 = st.columns(3)

    with col4:
        p_global = st.number_input(
            "PROM. PUNTAJE GLOBAL",
            min_value=0.0,
            max_value=500.0,
            value=float(df_base["PROM. PUNTAJE GLOBAL"].mean(skipna=True)),
        )
        p_ing = st.number_input(
            "PROM. PUNTAJE INGLÉS",
            min_value=0.0,
            max_value=100.0,
            value=float(df_base["PROM. PUNTAJE INGLÉS"].mean(skipna=True)),
        )
    with col5:
        p_mat = st.number_input(
            "PROM. PUNTAJE MATEMATICAS",
            min_value=0.0,
            max_value=100.0,
            value=float(df_base["PROM. PUNTAJE MATEMATICAS"].mean(skipna=True)),
        )
        p_lec = st.number_input(
            "PROM. PUNTAJE LECTURA CRITICA",
            min_value=0.0,
            max_value=100.0,
            value=float(df_base["PROM. PUNTAJE LECTURA CRITICA"].mean(skipna=True)),
        )
    with col6:
        p_soc = st.number_input(
            "PROM. PUNTAJE SOCIALES CIUDADANAS",
            min_value=0.0,
            max_value=100.0,
            value=float(df_base["PROM. PUNTAJE SOCIALES CIUDADANAS"].mean(skipna=True)),
        )
        p_cien = st.number_input(
            "PROM. PUNTAJE CIENCIAS NATURALES",
            min_value=0.0,
            max_value=100.0,
            value=float(df_base["PROM. PUNTAJE CIENCIAS NATURALES"].mean(skipna=True)),
        )

    if st.button("Calcular pensión y matrícula esperada"):
        nueva_raw = pd.DataFrame(
            {
                "DEPTO": [depto_sel],
                "MUNI": [muni_sel],
                "LOCALIDAD": [localidad_value],
                "TIPO DE MUNICIPIO": [tipo_muni_sel],
                "CALENDARIO": [calendario_sel],
                "ES_BILINGUE": [bilingue_sel],
                "JORNADA": [jornada_sel],
                "AÑOS DE OPERACIÓN": [anos_oper],
                "PLANTEL ICFES 2024": [icfes_sel],
                "PROM. PUNTAJE GLOBAL": [p_global],
                "PROM. PUNTAJE INGLÉS": [p_ing],
                "PROM. PUNTAJE MATEMATICAS": [p_mat],
                "PROM. PUNTAJE LECTURA CRITICA": [p_lec],
                "PROM. PUNTAJE SOCIALES CIUDADANAS": [p_soc],
                "PROM. PUNTAJE CIENCIAS NATURALES": [p_cien],
                "INSE": [inse],
                "NSE_ ESTABLECIMIENTO": [nse_estab],
                "NSE_ESTUDIANTE": [nse_estud],
                "ESTUDIANTES_TOTALES_2023": [est_tot],
            }
        )

        nueva_proc = preparar_nueva_muestra(nueva_raw)
        pension_pred = predecir_pension_ensemble(nueva_proc)[0]

        info = resumen_estudiantes_por_depto_y_pension(
            depto=depto_sel,
            muni=muni_sel,
            pension_pred=pension_pred,
            z=1.96,
        )

        colA, colB, colC = st.columns(3)
        colA.metric("Pensión estimada (modelo)", f"${pension_pred:,.0f}")
        colB.metric(
            "Rango de pensión similar (95%)",
            f"${info['pension_low']:,.0f}  -  ${info['pension_high']:,.0f}",
        )
        colC.metric(
            "Colegios comparables en el municipio",
            f"{info['n_colegios_muni_banda']}",
        )

        colD, colE = st.columns(2)
        colD.metric(
            "Prom. estudiantes en el departamento",
            f"{info['prom_est_dep']:.1f}",
        )
        if np.isnan(info["prom_est_muni_banda"]):
            colE.metric(
                "Prom. estudiantes (muni, banda pensión)",
                "No hay colegios en la banda",
            )
        else:
            colE.metric(
                "Prom. estudiantes (muni, banda pensión)",
                f"{info['prom_est_muni_banda']:.1f}",
            )

        # Distribución de pensión en el municipio
        st.markdown("### Distribuciones")
        col_hist1, col_hist2 = st.columns(2)

        pensiones_muni = df_base.loc[
            df_base["MUNI"] == muni_sel, "PENSIÓN"
        ]
        chart_pension = chart_hist_with_line(
            pensiones_muni,
            pension_pred,
            "Distribución de pensión en el municipio",
            "PENSIÓN",
        )
        if chart_pension is not None:
            col_hist1.altair_chart(chart_pension, use_container_width=True)
        else:
            col_hist1.info("No hay datos suficientes para la distribución de pensión.")

        # Distribución de estudiantes en colegios comparables
        if info["n_colegios_muni_banda"] > 0 and not np.isnan(
            info["prom_est_muni_banda"]
        ):
            est_comp = info["df_filtrado"]["ESTUDIANTES_TOTALES_2023"]
            chart_est = chart_hist_with_line(
                est_comp,
                info["prom_est_muni_banda"],
                "Distribución de estudiantes (colegios comparables)",
                "ESTUDIANTES_TOTALES_2023",
            )
            if chart_est is not None:
                col_hist2.altair_chart(chart_est, use_container_width=True)
            else:
                col_hist2.info(
                    "No hay datos suficientes para la distribución de estudiantes."
                )
        else:
            col_hist2.info(
                "No hay colegios comparables suficientes para graficar estudiantes."
            )

        st.markdown("### Colegios comparables en el municipio")

        if info["n_colegios_muni_banda"] == 0:
            st.info(
                "No se encontraron colegios en ese municipio con pensiones dentro de la banda de 95%."
            )
        else:
            df_show = info["df_filtrado"].copy()

            # Reordenar columnas para destacar nombre, DANE, localidad
            priority = []
            for col in [NAME_COL, DANE_COL, "DEPTO", "MUNI", "LOCALIDAD",
                        "PENSIÓN", "ESTUDIANTES_TOTALES_2023"]:
                if col and col in df_show.columns and col not in priority:
                    priority.append(col)
            other_cols = [c for c in df_show.columns if c not in priority]
            df_show = df_show[priority + other_cols]

            df_show = df_show.sort_values("PENSIÓN")
            st.dataframe(df_show)

            # Botón para descargar CSV con toda la información
            csv = df_show.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Descargar colegios comparables (CSV)",
                data=csv,
                file_name="colegios_comparables.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
