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
    import google.generativeai as genai
    from sklearn.metrics.pairwise import cosine_similarity
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
        prompt_satira_imagem: str = Field(..., description="Um prompt de sátira, baseado no conteúdo da notícia, para ser usado em um gerador de imagens. Deve ser criativo e com um tom humorístico ou irônico.")

    respostas = []
    for texto in articles_df['content']:
        print(f"Fazendo extração do {texto[:40]}...") # Irá aparecer no seu terminal
        while True:
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents = f"Extraia informacoes da noticia em texto cru dada a seguir: \n\n {texto}",
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": Noticia,
                    },
                )
                break
            except Exception as e:
                print(f"Erro na API: {e} \nTentando novamente em 3s...")
                time.sleep(3)
        respostas.append(response.text)

    lista_de_dicionarios = [json.loads(json_string or '{}') for json_string in respostas]
    processados_df = pd.DataFrame(lista_de_dicionarios)
    return processados_df

# ==============================================================================
# ==== FIM - SUAS FUNÇÕES GEMINI EXATAS ====
# ==============================================================================

# --- FUNÇÕES DE GERAÇÃO DE HTML (INTOCADAS) ---
def gerar_card_noticia(noticia: dict, idx: int) -> str:
    # ... seu código de card ...
    return f"""<div>...</div>"""

def gerar_html_newsletter(df: pd.DataFrame, interesse: str) -> str:
    # ... seu código de gerar HTML ...
    # Lembre-se de ter um placeholder para injetar os cards
    html_template = """<!DOCTYPE html>...</html>"""
    cards_html = ""
    for idx, row in df.iterrows():
        cards_html += gerar_card_noticia(row.to_dict(), idx)
    return html_template.replace("", cards_html)

# --- INTERFACE E WORKFLOW DO STREAMLIT ---
st.set_page_config(page_title="Gerador de Newsletter", layout="wide")
st.title("📰 Gerador de Newsletter com IA")

tema_busca = st.text_input("1. Tema geral para a busca", value="Inteligência Artificial")
interesse_ordem = st.text_input("2. Interesse específico para ordenar", value="IA na política, governo e prefeituras")
top_noticias = st.number_input("3. Quantidade de notícias para a newsletter", min_value=1, max_value=20, value=3)

if st.button("Gerar Newsletter", type="primary"):
    with st.status("Iniciando processo...", expanded=True) as status:
        st.session_state.status_bar = status
        
        status.update(label="Passo 1/5: Buscando notícias...")
        df_bruto = pega_noticias(tema_busca)

        if df_bruto.empty:
            st.error("Nenhuma notícia encontrada.")
            st.stop()

        status.update(label="Passo 2/5: Ordenando por relevância...")
        df_ordenado = ordenar_noticias_por_similaridade(interesse=interesse_ordem, df_noticias=df_bruto, top_n=top_noticias)

        status.update(label="Passo 3/5: Extraindo conteúdo...")
        df_com_conteudo = extrair_conteudo_noticias(df_ordenado)

        status.update(label="Passo 4/5: Processando com IA...")
        df_processado = processa_noticias_com_gemini(df_com_conteudo)

        status.update(label="Passo 5/5: Montando a newsletter...")
        df_com_conteudo.reset_index(drop=True, inplace=True)
        df_processado.reset_index(drop=True, inplace=True)
        df_final = pd.concat([df_com_conteudo, df_processado], axis=1)
        html_final = gerar_html_newsletter(df_final, interesse_ordem)
        
        status.update(label="Processo concluído!", state="complete", expanded=False)

    st.success("Newsletter gerada com sucesso!")
    # ... (código de exibição e download) ...


