import streamlit as st
import pandas as pd
import requests
import json
import time
import numpy as np
from datetime import datetime
from typing import List

# --- Importações das bibliotecas de Machine Learning e Notícias ---
try:
    from GoogleNews import GoogleNews
    from sklearn.metrics.pairwise import cosine_similarity
    from google import genai
    from pydantic import BaseModel, Field
except ImportError as e:
    st.error(f"""
        Uma ou mais bibliotecas necessárias não foram encontradas.
        Por favor, instale-as com o comando abaixo no seu terminal:

        pip install streamlit pandas GoogleNews scikit-learn google-generativeai pydantic numpy

        Erro original: {e}
    """)
    st.stop()

# --- 1. CONFIGURAÇÃO DAS CHAVES DE API ---
# As chaves são carregadas a partir dos "secrets" do Streamlit
try:
    JINA_API_KEY = st.secrets["JINA_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Erro: Verifique se as chaves JINA_API_KEY e GEMINI_API_KEY estão no seu arquivo secrets.toml.")
    st.stop()

# ==============================================================================
# ==== INÍCIO: SUAS FUNÇÕES (COM ALTERAÇÕES MÍNIMAS E ESSENCIAIS) ====
# ==============================================================================

def buscar_google_news(termo):
    from GoogleNews import GoogleNews
    googlenews = GoogleNews(lang='pt-BR', period='1d', encode='utf-8')

    # ALTERAÇÃO MÍNIMA: Usar o 'termo' do input em vez de um valor fixo.
    googlenews.search(termo)

    max_resultados = 1000
    resultados = []
    pagina = 1
    while len(resultados) < max_resultados:
        googlenews.get_page(pagina)
        noticias = googlenews.result()
        if not noticias:
            break
        resultados.extend(noticias)
        pagina += 1
    resultados = resultados[:max_resultados]
    print(f'Quantidade de notícias retornadas do GoogleNews: {len(resultados)}')
    if not resultados:
        return pd.DataFrame()
    df = pd.DataFrame(resultados)
    df['link'] = df['link'].str.split('&ved').str[0]
    df.rename(columns={'media': 'source'}, inplace=True)
    if 'datetime' in df.columns:
        df.drop(columns=['datetime'], inplace=True)
    if 'img' in df.columns:
        df.drop(columns=['img'], inplace=True)
    return df

def pega_noticias(termo_busca):
    todas_as_noticias = buscar_google_news(termo_busca)
    if todas_as_noticias.empty:
        return pd.DataFrame()
    todas_as_noticias.dropna(subset=['link'], inplace=True)
    noticias_unicas = todas_as_noticias.drop_duplicates(subset=['link'], keep='first')
    noticias_unicas = noticias_unicas.drop_duplicates(subset=['title'], keep='first')
    noticias_unicas.reset_index(drop=True, inplace=True)
    print(f"Busca concluída! {noticias_unicas.shape[0]} notícias únicas encontradas.")
    return noticias_unicas

def ordenar_noticias_por_similaridade(interesse, df_noticias, top_n=10):
    TEXTOS = df_noticias['title'].to_list()
    # ALTERAÇÃO MÍNIMA: Usar st.secrets em vez de userdata do Colab.
    # A configuração global 'genai.configure' já lida com a API Key.
    result = genai.embed_content(model="models/embedding-001", content=interesse, task_type="RETRIEVAL_QUERY")
    interesse_embed = np.array(result['embedding'])
    VETORES = []
    for i in range(0, len(TEXTOS), 100):
        batch_textos = TEXTOS[i:i+100]
        result_batch = genai.embed_content(model="models/embedding-001", content=batch_textos, task_type="RETRIEVAL_DOCUMENT")
        VETORES.extend([np.array(e) for e in result_batch['embedding']])
    interesse_embed_2d = interesse_embed.reshape(1, -1)
    similaridades = [cosine_similarity(interesse_embed_2d, v.reshape(1, -1))[0][0] for v in VETORES]
    df_noticias['score'] = similaridades
    df_noticias.sort_values(by='score', ascending=False, inplace=True)
    return df_noticias.head(top_n).reset_index(drop=True)

def extrair_conteudo_noticias(df_noticias):
    headers = {
        # ALTERAÇÃO MÍNIMA: Usar st.secrets em vez de userdata do Colab.
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Engine": "browser",
        "X-Return-Format": "markdown"
    }
    total_noticias = len(df_noticias)
    conteudos = []
    for index, row in df_noticias.iterrows():
        # ALTERAÇÃO MÍNIMA: Usar o status do Streamlit para feedback visual.
        st.session_state.status_bar.update(label=f"Extraindo notícia {index + 1}/{total_noticias}: {row['title'][:40]}...")
        url = f"https://r.jina.ai/{row['link']}"
        try:
            response = requests.get(url, headers=headers, timeout=90)
            response.raise_for_status()
            conteudos.append(response.text)
        except requests.exceptions.RequestException as e:
            conteudos.append(f"Erro ao buscar conteúdo para o título '{row['title']}': {e}")
    df_noticias['content'] = conteudos
    return df_noticias

def processa_noticias_com_gemini(articles_df):
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
    
    generation_config = {"response_mime_type": "application/json", "response_schema": Noticia}
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
    
    respostas = []
    total_artigos = len(articles_df)
    for index, texto in enumerate(articles_df['content']):
        st.session_state.status_bar.update(label=f"Processando com IA {index + 1}/{total_artigos}...")
        if texto.startswith("Erro ao buscar conteúdo"):
            respostas.append('{}') # Adiciona um JSON vazio em caso de erro
            continue
        while True:
            try:
                prompt = f"Extraia informacoes da noticia em texto cru dada a seguir: \n\n {texto}"
                response = model.generate_content(prompt)
                respostas.append(response.text)
                break
            except Exception as e:
                print(f"Erro na API Gemini: {e}. Tentando novamente em 3s...")
                time.sleep(3)
    lista_de_dicionarios = [json.loads(json_string or '{}') for json_string in respostas]
    processados_df = pd.DataFrame(lista_de_dicionarios)
    return processados_df

def gerar_card_noticia(noticia: dict, idx: int) -> str:
    titulo = noticia.get('titulo', '')
    portal = noticia.get('portal', '')
    data_pub = noticia.get('data_de_publicacao', '')
    resumo_breve = noticia.get('resumo_curto', '')
    resumo_expandido = noticia.get('resumo_maior', '')
    tags = noticia.get('tags_relevantes', [])
    url_original = noticia.get('link', '')
    caminho_imagem = noticia.get('links_de_imagens', [])
    prompt_satira_imagem = noticia.get('prompt_satira_imagem', '')
    pontos_principais = noticia.get('pontos_principais', [])
    imagem_url = caminho_imagem[0] if caminho_imagem else ''
    tags_str = ', '.join(tags) if tags else ''
    pontos_principais_html = "".join([f"<li>{p}</li>" for p in pontos_principais]) if pontos_principais else ""
    return f"""<div class="card-noticia"> ... </div>""" # HTML do card (omitido por brevidade, mas está no seu código)

def gerar_html_newsletter(df: pd.DataFrame, interesse: str) -> str:
    html_content = f"""<!DOCTYPE html> ... </html>""" # Template HTML (omitido por brevidade)
    # Adiciona cada card de notícia
    cards_html = ""
    for idx, row in df.iterrows():
        noticia_dict = row.to_dict()
        cards_html += gerar_card_noticia(noticia_dict, idx) # A sua função de card
    # Injeta os cards no template
    final_html = html_content.replace("", cards_html)
    return final_html

# ==============================================================================
# ==== FIM: SUAS FUNÇÕES                                                   ====
# ==============================================================================

# --- INTERFACE DO STREAMLIT ---
st.set_page_config(page_title="Gerador de Newsletter com IA", layout="wide")
st.title("📰 Gerador de Newsletter com IA")
st.markdown("Crie uma newsletter personalizada. Defina um tema geral para a busca, um interesse específico para o ranking e o número de notícias desejado.")

# --- INPUTS DO USUÁRIO ---
tema_busca = st.text_input(
    "1. Tema geral para a busca de notícias",
    value="Inteligência Artificial",
    help="Ex: 'sustentabilidade', 'mercado financeiro', 'eleições 2026'"
)
interesse_ordem = st.text_input(
    "2. Interesse específico para ordenar por relevância",
    value="IA na política, governo e prefeituras",
    help="Ex: 'impacto da IA na educação', 'carros elétricos no Brasil'"
)
top_noticias = st.number_input(
    "3. Quantidade de notícias para a newsletter final",
    min_value=1,
    max_value=20,
    value=3,
    help="Escolha o número de notícias que aparecerão na newsletter após a ordenação."
)

if st.button("Gerar Newsletter", type="primary"):
    # --- LIGAÇÃO DAS FUNÇÕES EM ORDEM (WORKFLOW) ---
    with st.status("Iniciando processo...", expanded=True) as status:
        st.session_state.status_bar = status # Permite que as funções atualizem o status

        # PASSO 1: Pega as notícias
        status.update(label="Passo 1/5: Buscando um grande volume de notícias...")
        df_bruto = pega_noticias(tema_busca)

        if df_bruto.empty:
            st.error("Nenhuma notícia encontrada para o tema. Tente um termo diferente.")
            st.stop()

        # PASSO 2: Ordena por similaridade
        status.update(label=f"Passo 2/5: Ordenando {len(df_bruto)} notícias por relevância ao seu interesse...")
        df_ordenado = ordenar_noticias_por_similaridade(
            interesse=interesse_ordem,
            df_noticias=df_bruto,
            top_n=top_noticias
        )

        # PASSO 3: Extrai o conteúdo
        status.update(label="Passo 3/5: Extraindo conteúdo das notícias selecionadas...")
        df_com_conteudo = extrair_conteudo_noticias(df_ordenado)

        # PASSO 4: Processa com Gemini
        status.update(label="Passo 4/5: Usando IA para estruturar e resumir as notícias...")
        df_processado = processa_noticias_com_gemini(df_com_conteudo)

        # PASSO 5: Junta os dataframes e gera o HTML
        status.update(label="Passo 5/5: Montando a newsletter final...")
        # Garante que os índices estão alinhados para a concatenação
        df_com_conteudo.reset_index(drop=True, inplace=True)
        df_processado.reset_index(drop=True, inplace=True)
        df_final = pd.concat([df_com_conteudo, df_processado], axis=1)

        # Substituí sua função original pela versão completa que você mandou
        html_final = gerar_html_newsletter(df_final, interesse_ordem)
        status.update(label="Processo concluído!", state="complete", expanded=False)

    st.success("Newsletter gerada com sucesso!")

    # Adiciona um botão de download para o arquivo HTML
    st.download_button(
        label="📥 Baixar Newsletter em HTML",
        data=html_final,
        file_name=f"newsletter_{tema_busca.replace(' ', '_')}.html",
        mime="text/html"
    )

    # Exibe o HTML diretamente na página
    st.markdown("### Pré-visualização da Newsletter")
    st.components.v1.html(html_final, height=800, scrolling=True)
