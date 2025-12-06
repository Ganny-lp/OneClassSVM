"""
oneclasssvm_app.py
Análise de outliers usando One-Class SVM e Isolation Forest.
Interface interativa com Streamlit com abas para cada método e análise comparativa.
"""

import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

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
# Funções utilitárias compartilhadas
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


# ---------------------------
# Funções específicas para One-Class SVM
# ---------------------------

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


# ---------------------------
# Funções específicas para Isolation Forest
# ---------------------------

def run_isolation_forest_analysis(pivot_table, contamination=0.1, n_estimators=100, random_state=42):
    """
    Executa análise de outliers usando Isolation Forest.
    """
    logger.info(f"Running Isolation Forest with contamination={contamination}, n_estimators={n_estimators}")
    
    # Preparar os dados para o modelo
    X = pivot_table.values
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Configurar e treinar o modelo Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        verbose=0
    )
    
    # Prever outliers
    outlier_predictions = iso_forest.fit_predict(X_scaled)
    
    # Adicionar previsões ao DataFrame
    pivot_table_result = pivot_table.copy()
    pivot_table_result['Is_Outlier'] = outlier_predictions
    pivot_table_result['Is_Outlier'] = pivot_table_result['Is_Outlier'].map({1: 'Normal', -1: 'Outlier'})
    
    # Criar DataFrame de resultados
    resultados_df = pivot_table_result.reset_index()
    resultados_df = resultados_df.rename(columns={'Unidade Orçamentária': 'Unidade Orçamentária'})
    
    return {
        'pivot_table_result': pivot_table_result,
        'resultados_df': resultados_df,
        'model': iso_forest,
        'X_scaled': X_scaled,
        'outlier_predictions': outlier_predictions
    }


# ---------------------------
# Funções para relatórios
# ---------------------------

def generate_report_md(results_df, pivot_table, years, out_file, method="One-Class SVM"):
    """
    Gera um relatório em Markdown com os resultados principais.
    """
    lines = []
    lines.append(f"# Relatório: {method} - Detecção de Outliers")
    lines.append(f"Data da análise: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    lines.append("## Sumário\n")
    lines.append(f"- Universidades analisadas: {pivot_table.shape[0]}")
    lines.append(f"- Anos considerados: {years[0]}–{years[-1]}\n")
    lines.append("## Resultados: classificação (amostra)\n")
    lines.append(results_df.head(50).to_markdown(index=False))
    lines.append("\n## Estatísticas por período (média investimento por aluno)\n")
    
    if method == "One-Class SVM":
        lines.append("\n## Notas metodológicas\n")
        lines.append(
            "- Pré-processamento: pivot por universidade x ano; valores faltantes preenchidos com 0.\n")
        lines.append(
            "- Heurística de seleção de hiperparâmetros: buscamos parâmetros do One-Class SVM cujo 'outlier_rate' aproximasse a contaminação alvo (10%).\n")
    else:
        lines.append("\n## Notas metodológicas\n")
        lines.append(
            "- Pré-processamento: pivot por universidade x ano; valores faltantes preenchidos com 0.\n")
        lines.append(
            "- Algoritmo: Isolation Forest com contaminação fixa em 10% (alinhado ao relatório original).\n")
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
    logger.info(f"Report saved to {out_file}")


# ---------------------------
# Funções principais com cache
# ---------------------------

@st.cache_data
def compute_results_oneclass(_excel_path, _sheet_name, _contamination, _n_jobs):
    """
    Função principal que calcula os resultados do One-Class SVM e armazena em cache.
    """
    try:
        # Carregar e pré-processar dados
        df_clean, grouped, pivot_table, years = load_and_preprocess(_excel_path, _sheet_name)
        
        # Escalar dados
        X = pivot_table.values
        X_scaled, scaler = scale_data(X)
        
        # Definir grade de parâmetros
        nu_values = np.linspace(0.01, 0.3, 10)
        gamma_values = np.logspace(-3, 0, 8)
        param_grid = list(ParameterGrid({"nu": nu_values, "gamma": gamma_values}))
        
        # Buscar melhores parâmetros
        best, results_sorted = grid_search_params(
            X_scaled, 
            param_grid, 
            target_contamination=_contamination,
            n_jobs=int(_n_jobs)
        )
        
        # Usar o melhor modelo para previsões finais
        best_model = best['model']
        preds = best_model.predict(X_scaled)
        label_map = np.where(preds == 1, "Normal", "Outlier")
        pivot_table_result = pivot_table.copy()
        pivot_table_result['Is_Outlier'] = label_map
        
        # Criar DataFrame de resultados
        resultados_df = pivot_table_result.reset_index()
        resultados_df = resultados_df.rename(columns={'Unidade Orçamentária': 'Unidade Orçamentária'})
        
        return {
            'success': True,
            'df_clean': df_clean,
            'grouped': grouped,
            'pivot_table': pivot_table,
            'pivot_table_result': pivot_table_result,
            'resultados_df': resultados_df,
            'years': years,
            'best_params': best,
            'X_scaled': X_scaled
        }
        
    except Exception as e:
        logger.exception("Erro ao computar resultados One-Class SVM")
        return {
            'success': False,
            'error': str(e)
        }


@st.cache_data
def compute_results_isolation_forest(_excel_path, _sheet_name, _contamination, _n_estimators, _random_state):
    """
    Função principal que calcula os resultados do Isolation Forest e armazena em cache.
    """
    try:
        # Carregar e pré-processar dados
        df_clean, grouped, pivot_table, years = load_and_preprocess(_excel_path, _sheet_name)
        
        # Executar análise Isolation Forest
        results = run_isolation_forest_analysis(
            pivot_table, 
            contamination=_contamination,
            n_estimators=int(_n_estimators),
            random_state=int(_random_state)
        )
        
        return {
            'success': True,
            'df_clean': df_clean,
            'grouped': grouped,
            'pivot_table': pivot_table,
            'pivot_table_result': results['pivot_table_result'],
            'resultados_df': results['resultados_df'],
            'years': years,
            'model': results['model'],
            'X_scaled': results['X_scaled']
        }
        
    except Exception as e:
        logger.exception("Erro ao computar resultados Isolation Forest")
        return {
            'success': False,
            'error': str(e)
        }


# ---------------------------
# Funções para visualizações
# ---------------------------

def plot_comparison_charts(resultados_df, pivot_table_result, years, method_name):
    """
    Gera gráficos de comparação para um método específico.
    """
    st.subheader(f"📈 Evolução Temporal: Comparação Outliers vs Normais ({method_name})")
    
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
                                title=f'Média de Investimento por Aluno: Outliers vs Normais ({method_name})',
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
    st.markdown("*Agora você pode selecionar até 6 universidades de cada tipo para comparação.*")
    
    # Verificar se temos dados
    if resultados_df is None or len(resultados_df) == 0:
        st.warning("Nenhum resultado disponível.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Selecionar Outliers**")
            outliers_list = resultados_df[resultados_df['Is_Outlier'] == "Outlier"]['Unidade Orçamentária'].tolist()
            if outliers_list:
                selected_outliers = st.multiselect(
                    "Outliers para comparar (máx 6)",
                    options=outliers_list,
                    default=outliers_list[:min(6, len(outliers_list))],
                    max_selections=6,
                    key=f"outliers_select_{method_name}"
                )
            else:
                st.info("Nenhum outlier detectado")
                selected_outliers = []
        
        with col2:
            st.markdown("**Selecionar Normais**")
            normals_list = resultados_df[resultados_df['Is_Outlier'] == "Normal"]['Unidade Orçamentária'].tolist()
            if normals_list:
                selected_normals = st.multiselect(
                    "Normais para comparar (máx 6)",
                    options=normals_list,
                    default=normals_list[:min(6, len(normals_list))],
                    max_selections=6,
                    key=f"normals_select_{method_name}"
                )
            else:
                st.info("Nenhuma universidade normal detectada")
                selected_normals = []
        
        # Gráfico de comparação - COM VERIFICAÇÃO ROBUSTA
        selected_all = selected_outliers + selected_normals
        
        if selected_all:
            comparison_data = []
            for uni in selected_all:
                row = resultados_df[resultados_df['Unidade Orçamentária'] == uni]
                if not row.empty:
                    for year in years:
                        valor = row[years].iloc[0][year]
                        comparison_data.append({
                            "Universidade": uni,
                            "Ano": year,
                            "Investimento_por_Aluno": valor,
                            "Classificacao": row['Is_Outlier'].iloc[0],
                            "Tipo_Exibicao": f"{uni} ({row['Is_Outlier'].iloc[0]})"
                        })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Criar gráfico com cores por classificação e estilo por universidade
                fig_comparison = px.line(comparison_df, x='Ano', y='Investimento_por_Aluno',
                                         color='Classificacao',
                                         line_dash='Universidade',
                                         markers=True,
                                         title=f'Comparação Direta: Evolução de Universidades Selecionadas ({method_name})',
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
            else:
                st.info("Selecione universidades válidas para visualizar a comparação.")
        else:
            st.info("Selecione pelo menos uma universidade para visualizar a comparação.")


# ---------------------------
# Aba de Análise Comparativa
# ---------------------------

def comparative_analysis(results_oneclass, results_if):
    """
    Realiza análise comparativa entre os dois métodos.
    """
    st.header("🔍 Análise Comparativa: One-Class SVM vs Isolation Forest")
    
    if not results_oneclass['success'] or not results_if['success']:
        st.error("É necessário executar ambas as análises antes de fazer a comparação.")
        return
    
    # Extrair resultados
    resultados_oc = results_oneclass['resultados_df']
    resultados_if = results_if['resultados_df']
    years = results_oneclass['years']
    
    # Criar DataFrame comparativo
    comparative_df = pd.DataFrame({
        'Unidade Orçamentária': resultados_oc['Unidade Orçamentária'],
        'OneClass_SVM': resultados_oc['Is_Outlier'],
        'Isolation_Forest': resultados_if['Is_Outlier']
    })
    
    # Adicionar coluna de concordância
    comparative_df['Concordância'] = comparative_df['OneClass_SVM'] == comparative_df['Isolation_Forest']
    
    # Calcular estatísticas
    total_universidades = len(comparative_df)
    concordantes = comparative_df['Concordância'].sum()
    discordantes = total_universidades - concordantes
    taxa_concordancia = concordantes / total_universidades * 100
    
    st.subheader("📊 Estatísticas de Concordância")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Universidades", total_universidades)
    with col2:
        st.metric("Concordantes", concordantes, f"{taxa_concordancia:.1f}%")
    with col3:
        st.metric("Discordantes", discordantes, f"{(100 - taxa_concordancia):.1f}%")
    
    # Matriz de confusão
    st.subheader("🤝 Matriz de Concordância entre Métodos")
    
    # Criar tabela de contingência
    contingency_table = pd.crosstab(
        comparative_df['OneClass_SVM'],
        comparative_df['Isolation_Forest'],
        margins=True,
        margins_name="Total"
    )
    
    # Renomear índices e colunas para melhor legibilidade
    contingency_table = contingency_table.rename(
        index={'Normal': 'SVM: Normal', 'Outlier': 'SVM: Outlier', 'Total': 'Total SVM'},
        columns={'Normal': 'IF: Normal', 'Outlier': 'IF: Outlier', 'Total': 'Total IF'}
    )
    
    st.dataframe(contingency_table.style.background_gradient(cmap='Blues'), use_container_width=True)
    
    # Análise dos discordantes
    st.subheader("⚠️ Universidades com Classificação Discordante")
    
    discordantes_df = comparative_df[~comparative_df['Concordância']].copy()
    
    if not discordantes_df.empty:
        # Adicionar dados de investimento para análise
        pivot_table = results_oneclass['pivot_table']
        
        discordant_data = []
        for _, row in discordantes_df.iterrows():
            uni = row['Unidade Orçamentária']
            oc_class = row['OneClass_SVM']
            if_class = row['Isolation_Forest']
            
            # Buscar dados de investimento
            uni_data = pivot_table.loc[uni] if uni in pivot_table.index else pd.Series([0]*len(years), index=years)
            
            # Calcular estatísticas
            media_investimento = uni_data.mean()
            var_percentual = ((uni_data.iloc[-1] - uni_data.iloc[0]) / uni_data.iloc[0] * 100) if uni_data.iloc[0] != 0 else 0
            
            discordant_data.append({
                'Universidade': uni,
                'OneClass_SVM': oc_class,
                'Isolation_Forest': if_class,
                'Média_Investimento': media_investimento,
                'Variação_%': var_percentual,
                'Ano_Pico': years[uni_data.argmax()] if len(uni_data) > 0 else None,
                'Valor_Pico': uni_data.max()
            })
        
        discordant_analysis_df = pd.DataFrame(discordant_data)
        
        # Ordenar por maior variação percentual
        discordant_analysis_df = discordant_analysis_df.sort_values('Variação_%', key=abs, ascending=False)
        
        st.dataframe(discordant_analysis_df, use_container_width=True)
        
        # Gráfico de dispersão: média vs variação
        fig_scatter = px.scatter(
            discordant_analysis_df,
            x='Média_Investimento',
            y='Variação_%',
            color='OneClass_SVM',
            symbol='Isolation_Forest',
            hover_name='Universidade',
            title='Universidades Discordantes: Média de Investimento vs Variação Percentual',
            labels={
                'Média_Investimento': 'Média de Investimento por Aluno (R$)',
                'Variação_%': 'Variação Percentual (2017-2024)',
                'OneClass_SVM': 'Classificação SVM',
                'Isolation_Forest': 'Classificação IF'
            }
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Análise por padrão de discordância
        st.subheader("📈 Padrões de Discordância")
        
        patterns = discordantes_df.groupby(['OneClass_SVM', 'Isolation_Forest']).size().reset_index(name='Count')
        patterns['Pattern'] = patterns['OneClass_SVM'] + ' → ' + patterns['Isolation_Forest']
        
        fig_patterns = px.bar(
            patterns,
            x='Pattern',
            y='Count',
            color='Pattern',
            title='Distribuição dos Padrões de Discordância',
            labels={'Pattern': 'Padrão (SVM → IF)', 'Count': 'Número de Universidades'}
        )
        
        st.plotly_chart(fig_patterns, use_container_width=True)
        
        # Exportar dados discordantes
        csv_path = os.path.join(OUT_DIR, "universidades_discordantes.csv")
        discordant_analysis_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        st.success(f"Dados discordantes exportados: `{csv_path}`")
    else:
        st.success("🎉 Perfeita concordância entre os dois métodos!")
    
    # Análise temporal comparativa
    st.subheader("🕒 Evolução Temporal Comparativa")
    
    # Preparar dados para comparação temporal
    temporal_data = []
    
    for method_name, results in [("One-Class SVM", results_oneclass), ("Isolation Forest", results_if)]:
        resultados_df = results['resultados_df']
        pivot_table_result = results['pivot_table_result']
        
        for class_type in ["Outlier", "Normal"]:
            type_unis = resultados_df[resultados_df['Is_Outlier'] == class_type]['Unidade Orçamentária']
            if len(type_unis) > 0:
                type_pivot = pivot_table_result[pivot_table_result.index.isin(type_unis)]
                for year in years:
                    if not type_pivot.empty:
                        year_data = type_pivot[year]
                        if len(year_data) > 0:
                            temporal_data.append({
                                "Método": method_name,
                                "Classificacao": class_type,
                                "Ano": year,
                                "Media_Investimento": year_data.mean(),
                                "Qtd_Universidades": len(type_unis)
                            })
    
    if temporal_data:
        temporal_df = pd.DataFrame(temporal_data)
        
        # Gráfico comparativo
        fig_comparative = px.line(
            temporal_df,
            x='Ano',
            y='Media_Investimento',
            color='Método',
            line_dash='Classificacao',
            markers=True,
            title='Comparação Temporal: Média de Investimento por Método e Classificação',
            labels={'Media_Investimento': 'Média Investimento/Aluno (R$)'}
        )
        
        st.plotly_chart(fig_comparative, use_container_width=True)
    
    # Recomendações baseadas na análise
    st.subheader("💡 Recomendações e Considerações")
    
    with st.expander("Ver recomendações detalhadas"):
        st.markdown("""
        ### Análise dos Resultados:
        
        1. **Alta Concordância (>80%)**: 
           - As universidades classificadas de forma consistente provavelmente representam casos claros de padrões normais ou atípicos.
           - Recomenda-se focar nas universidades concordantes para políticas específicas.
        
        2. **Discordâncias Significativas**:
           - As universidades com classificação discordante merecem análise individual.
           - Considere fatores adicionais não capturados pelos modelos.
        
        3. **Padrão "Normal → Outlier"**:
           - Universidades classificadas como normais pelo SVM mas outliers pelo IF podem representar casos limítrofes.
           - Recomenda-se análise manual desses casos.
        
        4. **Padrão "Outlier → Normal"**:
           - Universidades classificadas como outliers pelo SVM mas normais pelo IF podem ter padrões sazonais ou específicos.
        
        ### Ações Recomendadas:
        - **Priorizar análise** nas universidades discordantes
        - **Validar manualmente** as classificações extremas
        - **Considerar contexto institucional** para decisões finais
        - **Documentar critérios** para futuras análises
        """)
    
    return comparative_df


# ---------------------------
# Streamlit App Principal
# ---------------------------

def run_streamlit_app():
    if not STREAMLIT_AVAILABLE:
        st.error("Streamlit não está instalado. Instale com: pip install streamlit")
        return
    
    st.set_page_config(layout="wide", page_title="Análise Comparativa de Outliers")
    st.title("🔍 Análise Comparativa de Outliers em Universidades")
    st.markdown("### Comparação entre One-Class SVM e Isolation Forest")
    
    # Inicializar estado da sessão
    if 'results_oneclass' not in st.session_state:
        st.session_state.results_oneclass = None
    if 'results_if' not in st.session_state:
        st.session_state.results_if = None
    
    # Sidebar com parâmetros gerais
    st.sidebar.header("📁 Configurações Gerais")
    excel_path = st.sidebar.text_input("Caminho do arquivo Excel", value="Dados Finais.xlsx")
    sheet = st.sidebar.text_input("Nome da sheet", value="RemoveDuplicatas")
    
    # Criar abas
    tab1, tab2, tab3 = st.tabs([
        "🔬 One-Class SVM", 
        "🌲 Isolation Forest", 
        "📊 Análise Comparativa"
    ])
    
    # ----------------------------------
    # ABA 1: One-Class SVM
    # ----------------------------------
    with tab1:
        st.header("One-Class SVM — Detecção de Outliers")
        
        st.sidebar.header("⚙️ Parâmetros One-Class SVM")
        contamination_oc = st.sidebar.slider("Contaminação esperada", min_value=0.01, max_value=0.3, 
                                           value=0.10, step=0.01, key="contamination_oc")
        n_jobs = st.sidebar.number_input("n_jobs (parallelização)", min_value=1, max_value=16, 
                                       value=4, step=1, key="n_jobs")
        run_oneclass = st.sidebar.button("Rodar análise One-Class SVM", key="run_oneclass")
        
        if run_oneclass:
            with st.spinner("Processando dados e ajustando modelo One-Class SVM..."):
                results = compute_results_oneclass(excel_path, sheet, contamination_oc, n_jobs)
                st.session_state.results_oneclass = results
            
            if not results['success']:
                st.error(f"Erro ao processar dados: {results['error']}")
            else:
                st.success("✅ Análise One-Class SVM concluída!")
                
                # Extrair resultados
                df_clean = results['df_clean']
                grouped = results['grouped']
                pivot_table = results['pivot_table']
                pivot_table_result = results['pivot_table_result']
                resultados_df = results['resultados_df']
                years = results['years']
                best_params = results['best_params']
                
                st.write("Dimensão pivot (universidade x anos):", pivot_table.shape)
                
                # Salvar resultados
                csv_out = os.path.join(OUT_DIR, "resultados_oneclasssvm_universidades.csv")
                resultados_df.to_csv(csv_out, index=False, encoding='utf-8-sig')
                st.success(f"Resultados salvos: `{csv_out}`")
                
                # Distribuição
                st.subheader("📊 Distribuição: Normais vs Outliers")
                counts = resultados_df['Is_Outlier'].value_counts().reset_index()
                counts.columns = ['Classificacao', 'Count']
                fig_counts = px.bar(counts, x='Classificacao', y='Count',
                                    title='Distribuição de Universidades: Normais vs Outliers (One-Class SVM)')
                st.plotly_chart(fig_counts, use_container_width=True)
                
                # Gráficos de evolução temporal
                plot_comparison_charts(resultados_df, pivot_table_result, years, "One-Class SVM")
                
                # Heatmap
                st.subheader("🔥 Heatmap (Outliers - One-Class SVM)")
                outliers_df = pivot_table_result[pivot_table_result['Is_Outlier'] == "Outlier"]
                if not outliers_df.empty:
                    outliers_data = outliers_df[years].copy()
                    outliers_norm = outliers_data.div(outliers_data.max(axis=1).replace(0, 1), axis=0)
                    heat_fig = go.Figure(data=go.Heatmap(
                        z=outliers_norm.values,
                        x=years,
                        y=outliers_norm.index,
                        colorscale='YlOrRd'
                    ))
                    heat_fig.update_layout(title='Heatmap: Outliers (One-Class SVM)',
                                           xaxis_title='Ano', yaxis_title='Universidade')
                    st.plotly_chart(heat_fig, use_container_width=True)
                
                # Boxplot
                st.subheader("📦 Boxplot: Investimento por Período")
                try:
                    fig_box = px.box(grouped, x='Período', y='Investimento_por_Aluno', points="outliers",
                                     title='Distribuição do Investimento por Aluno por Período Pandêmico')
                    st.plotly_chart(fig_box, use_container_width=True)
                except Exception as e:
                    st.warning(f"Dados agrupados não disponíveis para boxplot: {e}")
                
                # Exportar relatório
                report_md = os.path.join(OUT_DIR, "relatorio_oneclasssvm.md")
                results_export = resultados_df.reset_index(drop=True)
                generate_report_md(results_export, pivot_table_result, years, report_md, "One-Class SVM")
                st.markdown(f"📄 Relatório markdown gerado: `{report_md}`")
                
                # Detalhes técnicos
                with st.expander("🔧 Ver detalhes técnicos do modelo One-Class SVM"):
                    st.write(f"**Melhores parâmetros encontrados:**")
                    st.write(f"- nu: {best_params['nu']}")
                    st.write(f"- gamma: {best_params['gamma']}")
                    st.write(f"- Taxa de outliers: {best_params['outlier_rate']:.2%}")
                    st.write(f"- Score (diferença da contaminação alvo): {best_params['score']:.4f}")
        
        elif st.session_state.results_oneclass is not None and st.session_state.results_oneclass['success']:
            st.info("✅ Resultados do One-Class SVM disponíveis. Clique no botão 'Rodar análise One-Class SVM' para reprocessar.")
    
    # ----------------------------------
    # ABA 2: Isolation Forest
    # ----------------------------------
    with tab2:
        st.header("Isolation Forest — Detecção de Outliers")
        
        st.sidebar.header("⚙️ Parâmetros Isolation Forest")
        contamination_if = st.sidebar.slider("Contaminação", min_value=0.01, max_value=0.3, 
                                           value=0.10, step=0.01, key="contamination_if")
        n_estimators = st.sidebar.number_input("Número de estimadores", min_value=10, max_value=500, 
                                             value=100, step=10, key="n_estimators")
        random_state = st.sidebar.number_input("Random state", min_value=0, max_value=100, 
                                             value=42, step=1, key="random_state")
        run_if = st.sidebar.button("Rodar análise Isolation Forest", key="run_if")
        
        if run_if:
            with st.spinner("Processando dados e ajustando modelo Isolation Forest..."):
                results = compute_results_isolation_forest(excel_path, sheet, contamination_if, n_estimators, random_state)
                st.session_state.results_if = results
            
            if not results['success']:
                st.error(f"Erro ao processar dados: {results['error']}")
            else:
                st.success("✅ Análise Isolation Forest concluída!")
                
                # Extrair resultados
                df_clean = results['df_clean']
                grouped = results['grouped']
                pivot_table = results['pivot_table']
                pivot_table_result = results['pivot_table_result']
                resultados_df = results['resultados_df']
                years = results['years']
                
                st.write("Dimensão pivot (universidade x anos):", pivot_table.shape)
                
                # Salvar resultados
                csv_out = os.path.join(OUT_DIR, "resultados_isolationforest_universidades.csv")
                resultados_df.to_csv(csv_out, index=False, encoding='utf-8-sig')
                st.success(f"Resultados salvos: `{csv_out}`")
                
                # Distribuição
                st.subheader("📊 Distribuição: Normais vs Outliers")
                counts = resultados_df['Is_Outlier'].value_counts().reset_index()
                counts.columns = ['Classificacao', 'Count']
                fig_counts = px.bar(counts, x='Classificacao', y='Count',
                                    title='Distribuição de Universidades: Normais vs Outliers (Isolation Forest)')
                st.plotly_chart(fig_counts, use_container_width=True)
                
                # Gráficos de evolução temporal
                plot_comparison_charts(resultados_df, pivot_table_result, years, "Isolation Forest")
                
                # Heatmap
                st.subheader("🔥 Heatmap (Outliers - Isolation Forest)")
                outliers_df = pivot_table_result[pivot_table_result['Is_Outlier'] == "Outlier"]
                if not outliers_df.empty:
                    outliers_data = outliers_df[years].copy()
                    outliers_norm = outliers_data.div(outliers_data.max(axis=1).replace(0, 1), axis=0)
                    heat_fig = go.Figure(data=go.Heatmap(
                        z=outliers_norm.values,
                        x=years,
                        y=outliers_norm.index,
                        colorscale='YlOrRd'
                    ))
                    heat_fig.update_layout(title='Heatmap: Outliers (Isolation Forest)',
                                           xaxis_title='Ano', yaxis_title='Universidade')
                    st.plotly_chart(heat_fig, use_container_width=True)
                
                # Boxplot
                st.subheader("📦 Boxplot: Investimento por Período")
                try:
                    fig_box = px.box(grouped, x='Período', y='Investimento_por_Aluno', points="outliers",
                                     title='Distribuição do Investimento por Aluno por Período Pandêmico')
                    st.plotly_chart(fig_box, use_container_width=True)
                except Exception as e:
                    st.warning(f"Dados agrupados não disponíveis para boxplot: {e}")
                
                # Exportar relatório
                report_md = os.path.join(OUT_DIR, "relatorio_isolationforest.md")
                results_export = resultados_df.reset_index(drop=True)
                generate_report_md(results_export, pivot_table_result, years, report_md, "Isolation Forest")
                st.markdown(f"📄 Relatório markdown gerado: `{report_md}`")
                
                # Detalhes técnicos
                with st.expander("🔧 Ver detalhes técnicos do modelo Isolation Forest"):
                    st.write(f"**Parâmetros utilizados:**")
                    st.write(f"- Contaminação: {contamination_if}")
                    st.write(f"- Número de estimadores: {n_estimators}")
                    st.write(f"- Random state: {random_state}")
                    outlier_count = (resultados_df['Is_Outlier'] == "Outlier").sum()
                    total_count = len(resultados_df)
                    st.write(f"- Taxa de outliers encontrada: {outlier_count/total_count*100:.2f}% ({outlier_count}/{total_count})")
        
        elif st.session_state.results_if is not None and st.session_state.results_if['success']:
            st.info("✅ Resultados do Isolation Forest disponíveis. Clique no botão 'Rodar análise Isolation Forest' para reprocessar.")
    
    # ----------------------------------
    # ABA 3: Análise Comparativa
    # ----------------------------------
    with tab3:
        st.header("📊 Análise Comparativa: One-Class SVM vs Isolation Forest")
        
        if st.session_state.results_oneclass is None or st.session_state.results_if is None:
            st.warning("⚠️ É necessário executar ambas as análises antes de fazer a comparação.")
            st.info("Por favor, execute:")
            st.info("1. One-Class SVM (aba 1)")
            st.info("2. Isolation Forest (aba 2)")
        else:
            if not st.session_state.results_oneclass['success'] or not st.session_state.results_if['success']:
                st.error("Uma ou ambas as análises falharam. Por favor, execute novamente.")
            else:
                comparative_df = comparative_analysis(st.session_state.results_oneclass, st.session_state.results_if)
                
                # Exportar análise comparativa
                csv_path = os.path.join(OUT_DIR, "analise_comparativa_completa.csv")
                comparative_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                st.success(f"📤 Análise comparativa completa exportada: `{csv_path}`")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Notas")
    st.sidebar.markdown("""
    - **One-Class SVM**: Baseado em otimização de hiperparâmetros para atingir taxa de contaminação alvo.
    - **Isolation Forest**: Usa contaminação fixa (10%) conforme relatório original.
    - **Análise Comparativa**: Compara concordância entre métodos e identifica casos discordantes.
    """)
    
    st.sidebar.markdown("### 📁 Saídas")
    st.sidebar.markdown(f"""
    - Figuras: `{FIG_DIR}/`
    - Logs: `{LOG_DIR}/`
    - Resultados: `{OUT_DIR}/`
    """)


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
