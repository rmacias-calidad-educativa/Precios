
import numpy as np
import pandas as pd
import streamlit as st
import joblib

BASE_FILE = "251107 Base colegios pricing (1).xlsx"
MODEL_FILE = "modelo_pension_ensemble_y_banda.joblib"

st.set_page_config(
    page_title="Simulador de pensión y matrícula",
    layout="wide",
)

@st.cache_data
def load_base():
    df = pd.read_excel(BASE_FILE)
    return df

@st.cache_resource
def load_model_bundle():
    bundle = joblib.load(MODEL_FILE)
    return bundle

df_base = load_base()
bundle = load_model_bundle()

modelos_pension = bundle["modelos_pension"]
usar_log_pension = bundle["usar_log_pension"]
feat_pension = bundle["feat_pension"]
resumen_pension = bundle["resumen_pension"]
sigma_resid_log = bundle["sigma_resid_log"]
orden_icfes = bundle.get("orden_icfes", {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1})
umbral_nuevo = bundle.get("umbral_nuevo", 5)

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

def main():
    st.title("Simulador de pensión y matrícula esperada")
    st.markdown(
        "Este visualizador usa un modelo de **pensión** y, a partir de la banda de pensión, "
        "estima cuántos estudiantes suelen tener colegios similares en el mismo municipio."
    )

    deptos = sorted(df_base["DEPTO"].dropna().unique())
    depto_sel = st.selectbox("Departamento (DEPTO)", deptos)

    munis = sorted(df_base.loc[df_base["DEPTO"] == depto_sel, "MUNI"].dropna().unique())
    muni_sel = st.selectbox("Municipio (MUNI)", munis)

    tipos_muni = sorted(df_base["TIPO DE MUNICIPIO"].dropna().unique())
    calendarios = sorted(df_base["CALENDARIO"].dropna().unique())
    bilingues = sorted(df_base["ES_BILINGUE"].dropna().unique())
    jornadas = sorted(df_base["JORNADA"].dropna().unique())
    icfes_opts = ["A+", "A", "B", "C", "D"]

    st.subheader("Características del colegio")
    col1, col2, col3 = st.columns(3)

    with col1:
        tipo_muni_sel = st.selectbox("Tipo de municipio", tipos_muni)
        calendario_sel = st.selectbox("Calendario", calendarios)
        bilingue_sel = st.selectbox("¿Es bilingüe?", bilingues)
        jornada_sel = st.selectbox("Jornada", jornadas)

    with col2:
        anos_oper = st.number_input("Años de operación", min_value=0, max_value=50, value=5)
        icfes_sel = st.selectbox("Plantel ICFES 2024", icfes_opts)
        inse = st.number_input("INSE", value=float(df_base["INSE"].mean(skipna=True)))
        nse_estab = st.number_input(
            "NSE establecimiento", value=float(df_base["NSE_ ESTABLECIMIENTO"].mean(skipna=True))
        )

    with col3:
        nse_estud = st.number_input(
            "NSE estudiante", value=float(df_base["NSE_ESTUDIANTE"].mean(skipna=True))
        )
        est_tot = st.number_input(
            "Estudiantes totales estimados",
            min_value=1.0,
            value=float(df_base["ESTUDIANTES_TOTALES_2023"].median(skipna=True)),
        )
        localidad_sel = st.text_input(
            "Localidad (opcional, solo grandes ciudades)", value=""
        )

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
                "LOCALIDAD": [localidad_sel if localidad_sel.strip() != "" else np.nan],
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

        st.markdown("### Colegios comparables en el municipio")
        if info["n_colegios_muni_banda"] == 0:
            st.info(
                "No se encontraron colegios en ese municipio con pensiones dentro de la banda de 95%."
            )
        else:
            cols_to_show = [
                "DEPTO",
                "MUNI",
                "PENSIÓN",
                "ESTUDIANTES_TOTALES_2023",
                "CALENDARIO",
                "ES_BILINGUE",
                "JORNADA",
            ]
            cols_to_show = [c for c in cols_to_show if c in info["df_filtrado"].columns]
            st.dataframe(info["df_filtrado"][cols_to_show].sort_values("PENSIÓN"))

if __name__ == "__main__":
    main()
