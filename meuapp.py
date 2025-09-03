import streamlit as st
import pandas as pd
import requests
import json
import time
import numpy as np
from datetime import datetime
from typing import List

# --- Importações das bibliotecas ---
try:
    from GoogleNews import GoogleNews
    from sklearn.metrics.pairwise import cosine_similarity
    from google import genai
    from google.generativeai import types
    from pydantic import BaseModel, Field
except ImportError as e:
    st.error(f"Erro de importação: {e}. Verifique se todas as bibliotecas estão na versão correta.")
    st.stop()

# --- Configuração das Chaves de API ---
try:
    JINA_API_KEY = st.secrets["JINA_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    # A configuração global não é usada pelas suas funções, mas é boa prática mantê-la.
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Erro: Verifique se JINA_API_KEY e GEMINI_API_KEY estão no seu arquivo secrets.toml.")
    st.stop()

# ==============================================================================
# ==== FUNÇÕES NÃO-GEMINI (INTOCADAS) ====
# ==============================================================================

def buscar_google_news(termo):
    googlenews = GoogleNews(lang='pt-BR', period='1d', encode='utf-8')
    googlenews.search(termo)
    max_resultados = 1000
    resultados = []
    pagina = 1
    while len(resultados) < max_resultados:
        googlenews.get_page(pagina)
        noticias = googlenews.result()
        if not noticias: break
        resultados.extend(noticias)
        pagina += 1
    resultados = resultados[:max_resultados]
    if not resultados: return pd.DataFrame()
    df = pd.DataFrame(resultados)
    df['link'] = df['link'].str.split('&ved').str[0]
    df.rename(columns={'media': 'source'}, inplace=True)
    if 'datetime' in df.columns: df.drop(columns=['datetime'], inplace=True)
    if 'img' in df.columns: df.drop(columns=['img'], inplace=True)
    return df

def pega_noticias(termo_busca):
    todas_as_noticias = buscar_google_news(termo_busca)
    if todas_as_noticias.empty: return pd.DataFrame()
    todas_as_noticias.dropna(subset=['link'], inplace=True)
    noticias_unicas = todas_as_noticias.drop_duplicates(subset=['link'], keep='first')
    noticias_unicas = noticias_unicas.drop_duplicates(subset=['title'], keep='first')
    noticias_unicas.reset_index(drop=True, inplace=True)
    return noticias_unicas

def extrair_conteudo_noticias(df_noticias):
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "X-Engine": "browser", "X-Return-Format": "markdown"}
    total_noticias = len(df_noticias)
    conteudos = []
    for index, row in df_noticias.iterrows():
        st.session_state.status_bar.update(label=f"Extraindo notícia {index + 1}/{total_noticias}...")
        url = f"https://r.jina.ai/{row['link']}"
        try:
            response = requests.get(url, headers=headers, timeout=90)
            response.raise_for_status()
            conteudos.append(response.text)
        except requests.exceptions.RequestException as e:
            conteudos.append(f"Erro ao buscar conteúdo: {e}")
    df_noticias['content'] = conteudos
    return df_noticias

# ==============================================================================
# ==== INÍCIO - SUAS FUNÇÕES GEMINI EXATAS ====
# ==============================================================================

def ordenar_noticias_por_similaridade(interesse, df_noticias, top_n=10):
    # SUA FUNÇÃO EXATA. A ÚNICA MUDANÇA É 'userdata.get' -> 'st.secrets'.
    TEXTOS = df_noticias['title'].to_list()

    client = genai.Client(api_key = st.secrets['GEMINI_API_KEY'])

    result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=interesse)

    interesse_embed  = np.array(result.embeddings[0].values)

    VETORES = []
    for i in range(0, len(TEXTOS), 100):
        batch_textos = TEXTOS[i:i+100]
        embeddings_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch_textos,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        ).embeddings
        VETORES.extend([np.array(e.values) for e in embeddings_result])

    interesse_embed_2d = interesse_embed.reshape(1, -1)
    similaridades = [cosine_similarity(interesse_embed_2d, v.reshape(1, -1))[0][0] for v in VETORES]
    df_noticias['score'] = similaridades
    df_noticias.sort_values(by='score', ascending=False, inplace=True)
    return df_noticias.head(top_n).reset_index(drop=True)


def processa_noticias_com_gemini(articles_df):
    # SUA FUNÇÃO EXATA. NENHUMA MUDANÇA.
    client = genai.Client(api_key = GEMINI_API_KEY)

    class Noticia(BaseModel):
        titulo: str = Field(..., description="O título da notícia.")
        data_de_publicacao: str = Field(..., description="A data em que a notícia foi publicada. Use sempre o formato: 'DD/MM/AAAA'.")
        autor: str = Field(..., description="O nome do autor da notícia.")
        portal: str = Field(..., description="O nome do portal de notícias onde a notícia foi publicada.")
        resumo_curto: str = Field(..., description="Um resumo conciso da notícia em torno de 50 palavras. De preferência para colocar informação adicional ao titulo (nao repetir a informacao do titulo)")
        resumo_maior: str = Field(..., description="Um resumo mais detalhado da notícia em torno de 500 palavras.")
        pontos_principais: List[str] = Field(..., description="um resumo da noticia em formato de lista item a item")
        noticia_completa: str = Field(..., description="O texto completo da notícia.")
        links_de_imagens: List[str] = Field(..., description="Uma lista de URLs das imagens associadas à notícia. Considere apenas aquelas relevantes para a noticia. Descarte logos, divulgacoes, etc...")
        tags_relevantes: List[str] = Field(..., description="Uma lista de tags ou palavras-chave relevantes para a notícia.")
        prompt_satira_imagem: str = Field(..., description="Um prompt de sátira, baseado no conteúdo da notícia, para ser usado em um gerador de imagens. Deve ser criativo e com um tom humorístico ou ir
