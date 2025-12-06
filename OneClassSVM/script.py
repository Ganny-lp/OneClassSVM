"""
oneclasssvm_app.py
Análise de outliers usando One-Class SVM (versão análoga ao Isolation Forest do relatório).
Interface interativa com Streamlit, logging estruturado em JSON, paralelização para busca de hiperparâmetros,
e geração de figuras e relatório.
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from joblib import Parallel, delayed

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import ParameterGrid

import plotly.express as px
import plotly.graph_objects as go

# Verificar se o Streamlit está disponível antes de importar
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit não está instalado. Instale com: pip install streamlit")

# ---------------------------
# Configuração de diretórios
# ---------------------------
BASE_DIR = os.getcwd()
FIG_DIR = os.path.join(BASE_DIR, "figuras")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUT_DIR = os.path.join(BASE_DIR, "resultados")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# Logging estruturado (JSON)
# ---------------------------
logger = logging.getLogger("oneclasssvm")
logger.setLevel(logging.DEBUG)
log_path = os.path.join(LOG_DIR, "oneclasssvm.log")


# JSON formatter
class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {
            "time": datetime.utcnow().isoformat() + "Z",
            "name": record.name,
            "level": record.levelname,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


fh = logging.handlers.RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(JsonFormatter())

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)


# ---------------------------
# Funções utilitárias
# ---------------------------

def load_and_preprocess(excel_path: str, sheet_name: str = "RemoveDuplicatas"):
    """
    Carrega o Excel e aplica o mesmo pré-processamento usado no relatório:
    - Conversões, remoção de NAs relevantes,
    - Exclude universidade RONDONOPOLIS,
    - Agrupamento por Unidade Orçamentária x Ano e cálculo Investimento_por_Aluno,
    - Pivot para formato (universidade x anos 2017-2024)
    """
    logger.info(f"Loading data from {excel_path} sheet={sheet_name}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    logger.debug(f"Loaded df shape: {df.shape}")

    # Conversões
    try:
        df['Ano Lançamento'] = df['Ano Lançamento'].astype(int)
    except Exception as e:
        logger.warning("Could not cast 'Ano Lançamento' to int; attempting coercion")
        df['Ano Lançamento'] = pd.to_numeric(df['Ano Lançamento'], errors='coerce').astype('Int64')

    df['Movim. Líquido - R$_destino'] = pd.to_numeric(df['Movim. Líquido - R$_destino'], errors='coerce')
    df['Quantidade Alunos'] = pd.to_numeric(df['Quantidade Alunos'], errors='coerce')

    # remover universidade RONDONOPOLIS (mesma regra do relatório)
    df = df[~df['Unidade Orçamentária'].str.contains('RONDONOPOLIS', case=False, na=False)]
    logger.info(f"After filtering RONDONOPOLIS shape: {df.shape}")

    # remover linhas sem valores essenciais
    df_clean = df.dropna(
        subset=['Movim. Líquido - R$_destino', 'Quantidade Alunos', 'Unidade Orçamentária', 'Ano Lançamento'])
    logger.info(f"After dropping invalid rows: {df_clean.shape}")

    # Agrupar por universidade e ano
    grouped = df_clean.groupby(['Unidade Orçamentária', 'Ano Lançamento']).agg({
        'Movim. Líquido - R$_destino': 'sum',
        'Quantidade Alunos': 'first',
        'Período': 'first'
    }).reset_index()

    grouped['Investimento_por_Aluno'] = grouped['Movim. Líquido - R$_destino'] / grouped['Quantidade Alunos']
    logger.info(f"Grouped dataset shape: {grouped.shape}")

    # Pivot (linhas: universidade; colunas: anos 2017..2024)
    years = list(range(2017, 2025))
    pivot_table = grouped.pivot_table(
        index='Unidade Orçamentária',
        columns='Ano Lançamento',
        values='Investimento_por_Aluno',
        aggfunc='mean'
    ).reindex(columns=years).fillna(0)

    logger.info(f"Pivot table shape: {pivot_table.shape}")
    return df_clean, grouped, pivot_table, years


def scale_data(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def evaluate_params_oneclass(X_scaled, nu, gamma, kernel='rbf', target_contamination=0.1):
    """
    Ajusta OneClassSVM com (nu, gamma) e retorna a diferença absoluta entre taxa de outliers encontrada
    e a target_contamination + outras métricas úteis.
    Observação: OneClassSVM predit -1 para outliers, 1 para inliers.
    """
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X_scaled)
    preds = model.predict(X_scaled)  # 1 normal, -1 outlier
    outlier_rate = (preds == -1).mean()
    score = abs(outlier_rate - target_contamination)
    return {
        "nu": nu,
        "gamma": gamma,
        "outlier_rate": outlier_rate,
        "score": score,
        "model": model,
        "preds": preds
    }


def grid_search_params(X_scaled, param_grid, target_contamination=0.1, n_jobs=4):
    """
    Paraleliza busca por hiperparâmetros. Retorna o best_result (menor diferença para contaminação alvo)
    e a lista de resultados.
    """
    logger.info(f"Starting grid search with {len(param_grid)} candidates (n_jobs={n_jobs})")

    # joblib Parallel approach
    def eval_candidate(params):
        return evaluate_params_oneclass(X_scaled, params['nu'], params['gamma'], params.get('kernel', 'rbf'),
                                        target_contamination)

    results = Parallel(n_jobs=n_jobs)(
        delayed(eval_candidate)(p) for p in param_grid
    )
    # ordenar por score ascendente
    results_sorted = sorted(results, key=lambda r: r['score'])
    best = results_sorted[0]
    logger.info(f"Best params found: nu={best['nu']}, gamma={best['gamma']}, outlier_rate={best['outlier_rate']:.4f}")
    return best, results_sorted


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.write_image(path, engine="kaleido")
    logger.info(f"Saved figure {path}")
    return path


def generate_report_md(results_df, pivot_table, years, out_file):
    """
    Gera um relatório em Markdown com os resultados principais.
    """
    lines = []
    lines.append("# Relatório: One-Class SVM - Detecção de Outliers")
    lines.append(f"Data da análise: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    lines.append("## Sumário\n")
    lines.append(f"- Universidades analisadas: {pivot_table.shape[0]}")
    lines.append(f"- Anos considerados: {years[0]}–{years[-1]}\n")
    lines.append("## Resultados: classificação (amostra)\n")
    lines.append(results_df.head(50).to_markdown(index=False))
    lines.append("\n## Estatísticas por período (média investimento por aluno)\n")
    # add brief stats
    lines.append("\n## Notas metodológicas\n")
    lines.append(
        "- Pré-processamento: pivot por universidade x ano; valores faltantes preenchidos com 0 (podem alterar conforme necessidade).\n")
    lines.append(
        "- Heurística de seleção de hiperparâmetros: buscamos parâmetros do One-Class SVM cujo 'outlier_rate' aproximasse a contaminação alvo (10%), alinhado à abordagem do Isolation Forest do relatório original.\n")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
    logger.info(f"Report saved to {out_file}")


# ---------------------------
# Streamlit App
# ---------------------------

def run_streamlit_app():
    if not STREAMLIT_AVAILABLE:
        st.error("Streamlit não está instalado. Instale com: pip install streamlit")
        return

    st.set_page_config(layout="wide", page_title="One-Class SVM - Detecção de Outliers")
    st.title("One-Class SVM — Análise de Outliers (Análogo ao Isolation Forest)")

    st.sidebar.header("Parâmetros")
    excel_path = st.sidebar.text_input("Caminho do arquivo Excel", value="Dados Finais.xlsx")
    sheet = st.sidebar.text_input("Nome da sheet", value="RemoveDuplicatas")
    contamination = st.sidebar.slider("Contaminação esperada (heurística)", min_value=0.01, max_value=0.3, value=0.10,
                                      step=0.01)
    n_jobs = st.sidebar.number_input("n_jobs (parallelização)", min_value=1, max_value=16, value=4, step=1)
    run_button = st.sidebar.button("Rodar análise")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Referência: resultados do Isolation Forest do relatório foram usados como guia metodológico e taxa de contaminação.")

    if not run_button:
        st.info("Ajuste parâmetros na barra lateral e clique em 'Rodar análise'.")
        return

    try:
        df_clean, grouped, pivot_table, years = load_and_preprocess(excel_path, sheet)
    except Exception as e:
        logger.exception("Erro ao carregar/pré-processar os dados")
        st.error(f"Erro ao carregar/pré-processar: {e}")
        return

    st.success("Dados carregados e pré-processados.")
    st.write("Dimensão pivot (universidade x anos):", pivot_table.shape)

    # Scale
    X = pivot_table.values
    X_scaled, scaler = scale_data(X)

    # Definir grade de parâmetros para busca do OneClassSVM
    # nu: fração estimada de outliers detectáveis -> [0.01, 0.5]
    # gamma: 'scale' ou grid numérico; aqui usamos grid numérico adaptado à variação dos dados
    nu_values = np.linspace(0.01, 0.3, 10)  # 10 valores
    # heurística para gamma: usar gama baseada na variância dos dados escalados
    gamma_values = np.logspace(-3, 0, 8)  # 8 valores
    param_grid = list(ParameterGrid({"nu": nu_values, "gamma": gamma_values}))

    st.info(f"Iniciando busca de hiperparâmetros ({len(param_grid)} candidatos) — isso pode levar alguns segundos.")
    logger.info("Launching grid search (parallel)")

    best, results_sorted = grid_search_params(X_scaled, param_grid, target_contamination=contamination,
                                              n_jobs=int(n_jobs))

    # Usar o melhor modelo para previsões finais
    best_model = best['model']
    preds = best_model.predict(X_scaled)  # 1 normal, -1 outlier
    label_map = np.where(preds == 1, "Normal", "Outlier")
    pivot_table_result = pivot_table.copy()
    pivot_table_result['Is_Outlier'] = label_map

    # Salvar resultados
    resultados_df = pivot_table_result.reset_index()
    resultados_df = resultados_df.rename(columns={'Unidade Orçamentária': 'Unidade Orçamentária'})
    csv_out = os.path.join(OUT_DIR, "resultados_oneclasssvm_universidades.csv")
    resultados_df.to_csv(csv_out, index=False, encoding='utf-8-sig')
    logger.info(f"Resultados salvos em {csv_out}")
    st.success(f"Resultados salvos: {csv_out}")

    # Mostrar contagem
    st.subheader("Distribuição: Normais vs Outliers")
    counts = resultados_df['Is_Outlier'].value_counts().reset_index()
    counts.columns = ['Classificacao', 'Count']
    fig_counts = px.bar(counts, x='Classificacao', y='Count',
                        title='Distribuição de Universidades: Normais vs Outliers')
    st.plotly_chart(fig_counts, use_container_width=True)

    # -----------------------------------------------------------
    # RELATÓRIO DE EVOLUÇÃO TEMPORAL - COMPARAÇÃO OUTLIERS vs NORMAIS
    # -----------------------------------------------------------

    st.subheader("📈 Evolução Temporal: Comparação Outliers vs Normais")

    # Primeiro gráfico: Visão geral por classificação (média por tipo)
    st.markdown("### 1. Média de Investimento por Tipo (Outlier vs Normal)")

    # Preparar dados para média por tipo
    type_data = []
    for class_type in ["Outlier", "Normal"]:
        type_unis = resultados_df[resultados_df['Is_Outlier'] == class_type]['Unidade Orçamentária']
        if len(type_unis) > 0:
            type_pivot = pivot_table_result[pivot_table_result.index.isin(type_unis)]
            for year in years:
                if not type_pivot.empty:
                    year_data = type_pivot[year]
                    if len(year_data) > 0:
                        type_data.append({
                            "Classificacao": class_type,
                            "Ano": year,
                            "Media_Investimento": year_data.mean(),
                            "Desvio_Padrao": year_data.std(),
                            "Qtd_Universidades": len(type_unis)
                        })

    if type_data:
        type_df = pd.DataFrame(type_data)

        # Gráfico de linha com média por tipo
        fig_type_mean = px.line(type_df, x='Ano', y='Media_Investimento', color='Classificacao',
                                markers=True,
                                title='Média de Investimento por Aluno: Outliers vs Normais (por ano)',
                                labels={'Media_Investimento': 'Média Investimento/Aluno (R$)',
                                        'Classificacao': 'Classificação'})

        # Adicionar área de desvio padrão
        for class_type in type_df['Classificacao'].unique():
            subset = type_df[type_df['Classificacao'] == class_type]
            fig_type_mean.add_traces([
                go.Scatter(
                    x=subset['Ano'].tolist() + subset['Ano'].tolist()[::-1],
                    y=(subset['Media_Investimento'] + subset['Desvio_Padrao']).tolist() +
                      (subset['Media_Investimento'] - subset['Desvio_Padrao']).tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)' if class_type == "Outlier" else 'rgba(0,0,255,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f'{class_type} ± desvio'
                )
            ])

        st.plotly_chart(fig_type_mean, use_container_width=True)

        # Estatísticas resumidas
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Estatísticas Outliers:**")
            outliers_df = resultados_df[resultados_df['Is_Outlier'] == "Outlier"]
            if not outliers_df.empty:
                outlier_stats = outliers_df[years].mean()
                st.write(f"Quantidade: {len(outliers_df)}")
                st.write(f"Média anual: {outlier_stats.mean():.2f} R$")
                st.write(f"Máximo: {outlier_stats.max():.2f} R$")

        with col2:
            st.markdown("**Estatísticas Normais:**")
            normals_df = resultados_df[resultados_df['Is_Outlier'] == "Normal"]
            if not normals_df.empty:
                normal_stats = normals_df[years].mean()
                st.write(f"Quantidade: {len(normals_df)}")
                st.write(f"Média anual: {normal_stats.mean():.2f} R$")
                st.write(f"Máximo: {normal_stats.max():.2f} R$")

    # Segundo gráfico: Comparação lado a lado
    st.markdown("### 2. Comparação Individual: Seleção de Universidades")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Selecionar Outliers**")
        outliers_list = resultados_df[resultados_df['Is_Outlier'] == "Outlier"]['Unidade Orçamentária'].tolist()
        selected_outliers = st.multiselect(
            "Outliers para comparar (máx 6)",
            options=outliers_list,
            default=outliers_list[:3] if len(outliers_list) >= 3 else outliers_list,
            max_selections=6,
            key="outliers_select"
        )

    with col2:
        st.markdown("**Selecionar Normais**")
        normals_list = resultados_df[resultados_df['Is_Outlier'] == "Normal"]['Unidade Orçamentária'].tolist()
        selected_normals = st.multiselect(
            "Normais para comparar (máx 6)",
            options=normals_list,
            default=normals_list[:3] if len(normals_list) >= 3 else normals_list,
            max_selections=6,
            key="normals_select"
        )

    # Gráfico de comparação
    selected_all = selected_outliers + selected_normals
    if selected_all:
        comparison_data = []
        for uni in selected_all:
            row = resultados_df[resultados_df['Unidade Orçamentária'] == uni]
            if not row.empty:
                for year in years:
                    comparison_data.append({
                        "Universidade": uni,
                        "Ano": year,
                        "Investimento_por_Aluno": row[years].iloc[0][year],
                        "Classificacao": row['Is_Outlier'].iloc[0],
                        "Tipo_Exibicao": f"{uni} ({row['Is_Outlier'].iloc[0]})"
                    })

        comparison_df = pd.DataFrame(comparison_data)

        # Criar gráfico com cores por classificação e estilo por universidade
        fig_comparison = px.line(comparison_df, x='Ano', y='Investimento_por_Aluno',
                                 color='Classificacao',
                                 line_dash='Universidade',
                                 markers=True,
                                 title='Comparação Direta: Evolução de Universidades Selecionadas',
                                 labels={'Investimento_por_Aluno': 'Investimento/Aluno (R$)',
                                         'Classificacao': 'Classificação'},
                                 hover_name='Tipo_Exibicao')

        st.plotly_chart(fig_comparison, use_container_width=True)

        # Tabela de dados
        with st.expander("📊 Ver dados da comparação"):
            pivot_comparison = comparison_df.pivot_table(
                index=['Universidade', 'Classificacao'],
                columns='Ano',
                values='Investimento_por_Aluno'
            ).reset_index()
            st.dataframe(pivot_comparison, use_container_width=True)

    # Terceiro gráfico: Todos os outliers juntos
    st.markdown("### 3. Perfil Temporal de Todos os Outliers")

    outliers_all = resultados_df[resultados_df['Is_Outlier'] == "Outlier"]['Unidade Orçamentária'].tolist()
    if outliers_all:
        outliers_data = []
        for uni in outliers_all:
            row = resultados_df[resultados_df['Unidade Orçamentária'] == uni]
            if not row.empty:
                for year in years:
                    outliers_data.append({
                        "Universidade": uni,
                        "Ano": year,
                        "Investimento_por_Aluno": row[years].iloc[0][year]
                    })

        outliers_plot_df = pd.DataFrame(outliers_data)

        # Gráfico de linha para todos os outliers
        fig_outliers_all = px.line(outliers_plot_df, x='Ano', y='Investimento_por_Aluno',
                                   color='Universidade',
                                   markers=True,
                                   title='Evolução de Todos os Outliers Detectados',
                                   labels={'Investimento_por_Aluno': 'Investimento/Aluno (R$)'})

        st.plotly_chart(fig_outliers_all, use_container_width=True)

        # Estatísticas dos outliers
        st.markdown("**Padrões Observados nos Outliers:**")
        outlier_patterns = []
        for uni in outliers_all:
            row = resultados_df[resultados_df['Unidade Orçamentária'] == uni]
            if not row.empty:
                values = row[years].iloc[0].values
                pattern = {
                    "Universidade": uni,
                    "Média": np.mean(values),
                    "Variação (%)": ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0,
                    "Ano_Pico": years[np.argmax(values)],
                    "Valor_Pico": np.max(values)
                }
                outlier_patterns.append(pattern)

        patterns_df = pd.DataFrame(outlier_patterns)
        st.dataframe(patterns_df.sort_values('Valor_Pico', ascending=False), use_container_width=True)

    # -----------------------------------------------------------
    # HEATMAP E OUTRAS VISUALIZAÇÕES (mantidas do código original)
    # -----------------------------------------------------------

    # Heatmap dos outliers (normalizado por linha)
    st.subheader("Heatmap (Outliers)")
    outliers_df = pivot_table_result[pivot_table_result['Is_Outlier'] == "Outlier"]
    if not outliers_df.empty:
        outliers_data = outliers_df[years].copy()
        # Normalizar por linha
        outliers_norm = outliers_data.div(outliers_data.max(axis=1).replace(0, 1), axis=0)
        heat_fig = go.Figure(data=go.Heatmap(
            z=outliers_norm.values,
            x=years,
            y=outliers_norm.index,
            colorscale='YlOrRd'
        ))
        heat_fig.update_layout(title='Heatmap: Outliers (Investimento por Aluno normalizado por universidade)',
                               xaxis_title='Ano', yaxis_title='Universidade')
        st.plotly_chart(heat_fig, use_container_width=True)
    else:
        st.info("Nenhum outlier detectado com os parâmetros selecionados.")

    # Boxplot por período (usando grouped)
    st.subheader("Boxplot: Investimento por Período")
    try:
        fig_box = px.box(grouped, x='Período', y='Investimento_por_Aluno', points="outliers",
                         title='Distribuição do Investimento por Aluno por Período Pandêmico')
        st.plotly_chart(fig_box, use_container_width=True)
    except Exception as e:
        st.warning(f"Dados agrupados não disponíveis para boxplot: {e}")

    # Exportar relatório markdown
    report_md = os.path.join(OUT_DIR, "relatorio_oneclasssvm.md")
    results_export = resultados_df.reset_index(drop=True)
    generate_report_md(results_export, pivot_table_result, years, report_md)
    st.markdown(f"Relatório markdown gerado: `{report_md}` (você pode converter para PDF externamente)")

    # Salvar figuras estáticas (opcional)
    try:
        fig_counts.write_image(os.path.join(FIG_DIR, "distribuicao_oneclasssvm.png"), engine="kaleido")
        logger.info("Saved distribution figure.")
    except Exception as e:
        logger.warning(f"Could not save static figure (kaleido may be missing): {e}")

    st.success("Análise concluída. Consulte logs em `logs/oneclasssvm.log` para detalhes.")


# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_streamlit_app()
    else:
        print("=" * 60)
        print("Streamlit não está instalado!")
        print("Para rodar esta aplicação:")
        print("1. Instale o Streamlit: pip install streamlit")
        print("2. Execute: streamlit run oneclasssvm_app.py")

        print("=" * 60)
