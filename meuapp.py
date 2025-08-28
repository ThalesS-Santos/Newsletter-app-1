import streamlit as st
import pandas as pd
import requests
import json

# --- BIBLIOTECAS ADICIONADAS ---
try:
    from newsdataapi import NewsDataApiClient
    import google.generativeai as genai
    from pydantic import BaseModel, Field
    from typing import List
    from serpapi import GoogleSearch
    from newsapi import NewsApiClient 

except ImportError as e:
    st.error(f"""
        Uma ou mais bibliotecas necessárias não foram encontradas.
        Por favor, instale-as executando o comando abaixo no seu terminal:
        
        pip install streamlit pandas requests newsdataapi google-generativeai pydantic serpapi-google-search newsapi-python

        Erro original: {e}
    """)
    st.stop()

# --- CHAVES DE API ATUALIZADAS ---
try:
    # Chaves existentes
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"] # Para NewsData.io
    JINA_API_KEY = st.secrets["JINA_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    # Novas chaves
    SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]
    NEWSAPI_ORG_KEY = st.secrets["NEWSAPI_ORG_KEY"]

    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("Erro: Uma ou mais chaves de API não foram encontradas. Verifique seu arquivo .streamlit/secrets.toml.")
    st.stop()

# --- NOVAS FUNÇÕES DE BUSCA (ADAPTADAS DO SEU CÓDIGO) ---
COLUNAS_FINAIS = ['title', 'link', 'source']

def buscar_newsdata(termo):
    try:
        api = NewsDataApiClient(apikey=NEWS_API_KEY)
        response = api.latest_api(q=termo, language='pt', country='br')
        resultados = response.get('results', [])
        if not resultados: return pd.DataFrame(columns=COLUNAS_FINAIS)
        df = pd.DataFrame(resultados)
        if 'title' in df.columns and 'link' in df.columns:
            df['source'] = 'NewsData.io'
            return df[COLUNAS_FINAIS]
        return pd.DataFrame(columns=COLUNAS_FINAIS)
    except Exception as e:
        st.warning(f"Erro ao buscar no NewsData.io: {e}")
        return pd.DataFrame(columns=COLUNAS_FINAIS)

def buscar_google_news(termo):
    try:
        params = {"q": termo, "tbm": "nws", "api_key": SERPAPI_API_KEY, "gl": "br", "hl": "pt-br"}
        search = GoogleSearch(params)
        results = search.get_dict()
        noticias = results.get('news_results', [])
        if not noticias: return pd.DataFrame(columns=COLUNAS_FINAIS)
        df = pd.DataFrame(noticias)
        if 'title' in df.columns and 'link' in df.columns:
            df['source'] = 'Google News'
            return df[COLUNAS_FINAIS]
        return pd.DataFrame(columns=COLUNAS_FINAIS)
    except Exception as e:
        st.warning(f"Erro ao buscar no Google News: {e}")
        return pd.DataFrame(columns=COLUNAS_FINAIS)

def buscar_google_search(termo):
    try:
        params = {"q": termo, "api_key": SERPAPI_API_KEY, "gl": "br", "hl": "pt-br"}
        search = GoogleSearch(params)
        results = search.get_dict()
        noticias = [res for res in results.get('organic_results', []) if 'title' in res and 'link' in res]
        if not noticias: return pd.DataFrame(columns=COLUNAS_FINAIS)
        df = pd.DataFrame(noticias)
        if 'title' in df.columns and 'link' in df.columns:
            df['source'] = 'Google Search'
            return df[COLUNAS_FINAIS]
        return pd.DataFrame(columns=COLUNAS_FINAIS)
    except Exception as e:
        st.warning(f"Erro ao buscar no Google Search: {e}")
        return pd.DataFrame(columns=COLUNAS_FINAIS)

def buscar_newsapi_org(termo):
    try:
        newsapi = NewsApiClient(api_key=NEWSAPI_ORG_KEY)
        response = newsapi.get_everything(q=termo, language='pt', sort_by='relevancy')
        noticias = response.get('articles', [])
        if not noticias: return pd.DataFrame(columns=COLUNAS_FINAIS)
        df = pd.DataFrame(noticias)
        df.rename(columns={'url': 'link'}, inplace=True)
        if 'title' in df.columns and 'link' in df.columns:
            df['source'] = 'NewsAPI.org'
            return df[COLUNAS_FINAIS]
        return pd.DataFrame(columns=COLUNAS_FINAIS)
    except Exception as e:
        st.warning(f"Erro ao buscar no NewsAPI.org: {e}")
        return pd.DataFrame(columns=COLUNAS_FINAIS)

# --- FUNÇÃO PRINCIPAL DE BUSCA (MODIFICADA) ---
@st.cache_data(ttl=3600)
def pega_noticias(termo_busca, max_noticias=5):
    """Busca notícias de múltiplas fontes, combina e remove duplicatas."""
    with st.spinner("Buscando em NewsData.io, Google News, Google Search e NewsAPI.org..."):
        lista_de_noticias_dfs = [
            buscar_newsdata(termo_busca),
            buscar_google_news(termo_busca),
            buscar_google_search(termo_busca),
            buscar_newsapi_org(termo_busca)
        ]

        todas_as_noticias = pd.concat(lista_de_noticias_dfs, ignore_index=True)
        
        if todas_as_noticias.empty:
            return pd.DataFrame()

        # Limpa e remove duplicatas baseadas no link
        todas_as_noticias.dropna(subset=['link'], inplace=True)
        noticias_unicas = todas_as_noticias.drop_duplicates(subset=['link'], keep='first')
        
        st.success(f"Busca concluída! {len(noticias_unicas)} notícias únicas encontradas (antes do limite).")
        return noticias_unicas.head(max_noticias)

# --- O RESTANTE DO CÓDIGO PERMANECE IGUAL ---
@st.cache_data(ttl=3600)
def extrair_conteudo_noticias(df_noticias):
    """Extrai o conteúdo completo dos artigos usando a Jina AI API."""
    conteudos = []
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Engine": "browser"
    }
    
    total_noticias = len(df_noticias)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, row in df_noticias.iterrows():
        status_text.text(f"Extraindo notícia {index + 1}/{total_noticias}: {row['title'][:50]}...")
        url = f"https://r.jina.ai/{row['link']}"
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            conteudos.append(response.text)
        except requests.exceptions.RequestException as e:
            conteudos.append(f"Erro ao buscar conteúdo para o título '{row['title']}': {e}")
        
        progress_bar.progress((index + 1) / total_noticias)
    
    status_text.empty()
    return pd.DataFrame({
        'title': df_noticias['title'],
        'link': df_noticias['link'],
        'content': conteudos
    })

@st.cache_data(ttl=3600)
def processa_noticias_com_gemini(df_conteudos):
    """Processa o conteúdo das notícias com a API do Gemini para extrair e estruturar dados."""
    
    class Noticia(BaseModel):
        titulo: str = Field(..., description="O título da notícia.")
        data_de_publicacao: str = Field(..., description="A data em que a notícia foi publicada (se disponível).")
        resumo_curto: str = Field(..., description="Um resumo conciso da notícia em até 30 palavras.")
        resumo_maior: str = Field(..., description="Um resumo mais detalhado da notícia em até 150 palavras.")
        links_de_imagens: List[str] = Field(..., description="Uma lista contendo até 2 URLs das imagens mais relevantes da notícia. Se não houver, retorne uma lista vazia.")

    respostas_json = []
    links_originais = df_conteudos['link'].tolist()

    total_conteudos = len(df_conteudos)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, texto in enumerate(df_conteudos['content']):
        status_text.text(f"Analisando com IA - Notícia {i + 1}/{total_conteudos}")
        if texto.startswith("Erro ao buscar conteúdo"):
            respostas_json.append(json.dumps({"titulo": "Conteúdo da notícia não disponível"}))
            progress_bar.progress((i + 1) / total_conteudos)
            continue
        
        try:
            model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite")
            response = model.generate_content(
                f"Analise o seguinte texto de uma notícia e extraia as informações no formato JSON, conforme o schema solicitado. Texto da notícia:\n\n---\n\n{texto}",
                generation_config={"response_mime_type": "application/json"},
                tools=[Noticia]
            )
            
            noticia_processada = json.loads(response.text)
            noticia_processada['link'] = links_originais[i]
            respostas_json.append(json.dumps(noticia_processada, ensure_ascii=False))

        except Exception as e:
            st.warning(f"Erro ao processar notícia com Gemini: {e}")
            respostas_json.append(json.dumps({"titulo": "Conteúdo da notícia não disponível"}))
        
        progress_bar.progress((i + 1) / total_conteudos)
    
    status_text.empty()
    return respostas_json


def gerar_newsletter_streamlit(lista_json):
    """Renderiza a newsletter na interface do Streamlit."""
    if not lista_json:
        st.info("Nenhuma notícia processada para exibir.")
        return
    
    st.write(f"**Exibindo {len(lista_json)} notícias processadas:**")

    for i, noticia_str in enumerate(lista_json):
        try:
            noticia = json.loads(noticia_str)
            if not noticia or noticia.get("titulo") == "Conteúdo da notícia não disponível":
                continue
        except (json.JSONDecodeError, AttributeError):
            continue

        titulo = noticia.get("titulo", "Título não encontrado")
        data = noticia.get("data_de_publicacao", "Data não informada")
        resumo_curto = noticia.get("resumo_curto", "")
        resumo_maior = noticia.get("resumo_maior", "")
        link = noticia.get("link", "#")
        imagens = noticia.get("links_de_imagens", [])
        imagem = imagens[0] if imagens else "https://via.placeholder.com/400x267?text=Sem+Imagem"

        with st.container():
            col_img, col_content = st.columns([1, 4])
            with col_img:
                st.image(imagem, use_container_width='always')
            with col_content:
                st.subheader(titulo)
                st.caption(f"Publicado em: {data}")
                st.write(resumo_curto)
                if resumo_maior:
                    with st.expander("Ler mais..."):
                        st.write(resumo_maior)
                st.markdown(f'<a href="{link}" target="_blank">Notícia completa ↗</a>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)



st.set_page_config(page_title="Gerador de Newsletter com IA", layout="centered")

st.markdown("""
<style>
.st-emotion-cache-1r4qj8v { border: 1px solid #e6e6e6; border-radius: 10px; padding: 1rem 1rem 1rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }
.st-emotion-cache-1r4qj8v:hover { transform: translateY(-4px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
a { color: #0a9396 !important; font-weight: 500; text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

st.title("📰 Gerador de Newsletter com IA")
st.markdown("Digite um tema, clique em gerar e obtenha um resumo das últimas notícias de múltiplas fontes, processado por Inteligência Artificial.")

termo_busca = st.text_input("Qual tema você quer pesquisar?",)
max_noticias = st.number_input("Número máximo de notícias para a newsletter", min_value=1, max_value=20, value=5, help="Selecione o número de notícias para processar e exibir (máx. 20).")


if st.button("Gerar Newsletter"):
    if not termo_busca:
        st.warning("Por favor, digite um termo para a busca.")
    else:
        df_noticias = pega_noticias(termo_busca, max_noticias)

        if not df_noticias.empty:
            st.info(f"Iniciando processamento de {len(df_noticias)} notícias...")
            
            df_conteudos = extrair_conteudo_noticias(df_noticias)
            
            resumos_json = processa_noticias_com_gemini(df_conteudos)
            
            st.success("Newsletter gerada com sucesso!")
            st.markdown("---")
            gerar_newsletter_streamlit(resumos_json)
        else:
            st.error(f"Nenhuma notícia encontrada para o termo '{termo_busca}' em nenhuma das fontes. Tente outro termo.")




