import pandas as pd
import requests
import re
import time
import unicodedata
from tqdm import tqdm


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_ENTRADA = "books_corrigido.csv"

# ------------------------------------------------------------
# DEFINA AQUI QUAL LOTE SERÁ PROCESSADO
# ------------------------------------------------------------
#
# Lote 1 -> 0 até aproximadamente 2.224
# Lote 2 -> aproximadamente 2.225 até 4.449
# Lote 3 -> aproximadamente 4.450 até 6.674
# Lote 4 -> aproximadamente 6.675 até 8.899
# Lote 5 -> aproximadamente 8.900 até 11.126
#
# Você só precisa alterar este número entre as execuções.
#
LOTE = 5

TOTAL_LOTES = 5


URL_ISBN = "https://openlibrary.org/isbn/{}.json"
URL_SEARCH = "https://openlibrary.org/search.json"
URL_WORK = "https://openlibrary.org/works/{}.json"


HEADERS = {
    "User-Agent": "SimilarBooksTest/1.0 (projeto educacional)"
}


# ============================================================
# FUNÇÕES DE LIMPEZA
# ============================================================

def limpar_isbn(isbn):

    if pd.isna(isbn):
        return ""

    isbn = str(isbn).strip()

    return re.sub(r"[^0-9Xx]", "", isbn)


def limpar_titulo(titulo):

    if pd.isna(titulo):
        return ""

    titulo = str(titulo).strip()

    # Remove informações de série entre parênteses.
    titulo = re.sub(r"\s*\([^)]*\)", "", titulo)

    return titulo.strip()


def limpar_autor(autor):

    if pd.isna(autor):
        return ""

    autor = str(autor).strip()

    # O dataset separa autores utilizando "/".
    primeiro_autor = autor.split("/")[0]

    return primeiro_autor.strip()


def remover_acentos(texto):

    texto = str(texto)

    return "".join(
        caractere
        for caractere in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caractere) != "Mn"
    )


def normalizar_subject(subject):

    if pd.isna(subject):
        return ""

    s = str(subject).strip()

    # Remove espaços duplicados
    s = re.sub(r"\s+", " ", s)

    # Remove ponto final
    s = s.rstrip(".")

    # Remove idioma no final.
    #
    # Exemplo:
    # Fantasy, English
    # vira:
    # Fantasy
    #
    s = re.sub(
        r",\s*(english|spanish|portuguese|french|german|italian)$",
        "",
        s,
        flags=re.IGNORECASE
    )

    # Remove ", Fiction" quando aparece como classificação.
    s = re.sub(
        r",\s*fiction$",
        "",
        s,
        flags=re.IGNORECASE
    )

    return s.strip()


# ============================================================
# FILTRO DE GÊNEROS E TEMAS
# ============================================================

def eh_genero_ou_tema(subject):

    # --------------------------------------------------------
    # Normalização
    # --------------------------------------------------------

    s_original = normalizar_subject(subject)

    if not s_original:
        return False

    s = remover_acentos(s_original).lower().strip()


    # ========================================================
    # 1. CLASSIFICAÇÕES QUE NÃO SÃO TEMAS
    # ========================================================

    assuntos_exatos = {

        # Genéricos
        "fiction",
        "general",
        "literature",
        "books",

        # Idiomas
        "english",
        "spanish",
        "portuguese",
        "french",
        "german",
        "italian",

        # Juvenil
        "juvenile works",
        "juvenile literature",
        "juvenile fiction",
        "juvenile audience",

        # Infantil
        "children's books",
        "children's fiction",
        "children's literature",
        "children's stories",

        # Young Adult
        "young adult fiction",
        "young adult literature",

        # Pessoas
        "biography",
        "biographies",
        "authors",
        "author",

        # Personagens classificados como categoria
        "fictitious character",
        "fictitious characters",
        "fictional character",
        "fictional characters",

        # Catálogo
        "collectibles",
        "collectors",
        "collecting",
        "prices",
        "bibliography",
        "bibliographies",
        "catalogs",
        "catalogues",

        # Crítica
        "book reviews",
        "book review",
        "criticism",
        "study and teaching",

        # Prêmios
        "awards",
        "prizes",
        "winner",
        "winners",

        # Open Library
        "open library staff picks",

        # Classificações editoriais estrangeiras
        "romans",
        "roman",
        "magie",
        "magiciens",
        "magicien",
        "fiction",
        "litterature",
        "litterature jeunesse",
        "litterature enfantine",
    }

    if s in assuntos_exatos:
        return False


    # ========================================================
    # 2. PRÊMIOS E BESTSELLERS
    # ========================================================

    palavras_premio = [

        "award winner",
        "award winners",

        "book award",
        "book awards",

        "book prize",
        "book prizes",

        "prize winner",
        "prize winners",

        "bestseller",
        "best seller",

        "award",
        "prize",
    ]

    for palavra in palavras_premio:

        if palavra in s:
            return False


    # ========================================================
    # 3. NEW YORK TIMES / BESTSELLER
    # ========================================================

    if "new york times" in s:

        if "bestseller" in s:
            return False

        if "best seller" in s:
            return False


    # ========================================================
    # 4. PERSONAGENS FICTÍCIOS
    # ========================================================

    if "fictitious character" in s:
        return False

    if "fictitious characters" in s:
        return False

    if "fictional character" in s:
        return False

    if "fictional characters" in s:
        return False


    # ========================================================
    # 5. CLASSIFICAÇÕES JUVENIS
    # ========================================================

    if "juvenile" in s:
        return False

    if "children's books" in s:
        return False

    if "children's fiction" in s:
        return False

    if "children's literature" in s:
        return False

    if "children's stories" in s:
        return False

    if "young adult" in s:
        return False


    # ========================================================
    # 6. IDIOMAS E TRADUÇÕES
    # ========================================================

    if s.endswith(" language materials"):
        return False

    if "translation" in s:
        return False

    if "translations" in s:
        return False

    if "traduccion" in s:
        return False

    if "traducciones" in s:
        return False

    if s.startswith("translation"):
        return False

    if s.startswith("translations"):
        return False

    if s.startswith("traduccion"):
        return False

    if s.startswith("traducciones"):
        return False


    # ========================================================
    # 7. CLASSIFICAÇÕES EDITORIAIS EM ESPANHOL
    # ========================================================

    termos_espanhol = {

        "ficcion juvenil",
        "novela juvenil",
        "novela inglesa",
        "novela fantastica",
        "novela de ficcion",
        "literatura juvenil",

        "fiction juvenil",
        "fiction infantil",

        "traduccion al espanol",
        "traducciones al espanol",
        "traduccion al ingles",
        "traducciones al ingles",
    }

    if s in termos_espanhol:
        return False


    if s.startswith("novela juvenil"):
        return False

    if s.startswith("novela inglesa"):
        return False

    if s.startswith("novela fantastica"):
        return False

    if s.startswith("ficcion juvenil"):
        return False

    if s.startswith("literatura juvenil"):
        return False


    # ========================================================
    # 8. CLASSIFICAÇÕES EDITORIAIS EM FRANCÊS
    # ========================================================

    termos_frances = {

        "romans",
        "roman",
        "nouvelles",
        "magie",
        "magicien",
        "magiciens",
        "litterature",
        "litterature jeunesse",
        "litterature enfantine",
    }

    if s in termos_frances:
        return False


    if "pour la jeunesse" in s:
        return False

    if "pour enfants" in s:
        return False

    if "pour enfants et adolescents" in s:
        return False


    # ========================================================
    # 9. LOCALIZAÇÕES
    # ========================================================

    locais = {

        "england",
        "london",
        "united states",
        "america",
        "europe",
        "france",
        "germany",
        "italy",
        "spain",
        "scotland",
        "ireland",
        "new york",
        "california",
        "brazil",
        "canada",
        "australia",
        "asia",
        "africa",
    }

    if s in locais:
        return False

    for local in locais:

        if s.startswith(local + ","):
            return False


    # ========================================================
    # 10. PERÍODOS HISTÓRICOS
    # ========================================================

    periodos = {

        "century",
        "20th century",
        "19th century",
        "18th century",
        "17th century",
        "16th century",
        "15th century",
        "14th century",
        "13th century",
        "12th century",
        "11th century",
        "10th century",

        "medieval",
        "middle ages",
    }

    if s in periodos:
        return False

    if "century" in s:
        return False


    # ========================================================
    # 11. ANOS
    # ========================================================

    if re.fullmatch(r"\d{4}", s):
        return False


    # ========================================================
    # 12. ORGANIZAÇÕES IMAGINÁRIAS
    # ========================================================

    if "imaginary organization" in s:
        return False

    if "imaginary organizations" in s:
        return False


    # ========================================================
    # 13. SÉRIES
    # ========================================================

    if s.startswith("series:"):
        return False

    if s.startswith("series "):
        return False


    # ========================================================
    # 14. CLASSIFICAÇÕES EDITORIAIS
    # ========================================================

    if s.startswith("children's books/"):
        return False

    if s.startswith("children's fiction/"):
        return False

    if s.startswith("young adult/"):
        return False


    # ========================================================
    # 15. CATEGORIAS MUITO GENÉRICAS
    # ========================================================

    categorias_genericas = {

        "fantasy - general",
        "fiction - general",
        "fiction / general",

        "fiction/literature",
        "fiction / literature",

        "fiction - literature",
    }

    if s in categorias_genericas:
        return False


    # ========================================================
    # 16. CATEGORIAS COM GENERAL
    # ========================================================

    if " - general" in s:
        return False

    if " / general" in s:
        return False


    # ========================================================
    # 17. CATEGORIAS GENÉRICAS EM OUTROS IDIOMAS
    # ========================================================

    categorias_outros_idiomas = {

        "magia",
        "escuelas",
        "ficcion",
        "literatura",
        "novela",
        "novelas",
        "fantasia general",

        "roman",
        "romans",
        "litterature",
    }

    if s in categorias_outros_idiomas:
        return False


    # ========================================================
    # 18. ASSUNTO APROVADO
    # ========================================================

    return True


# ============================================================
# EXTRAIR ASSUNTOS
# ============================================================

def extrair_assuntos(dados):

    assuntos = []

    if not dados:
        return assuntos

    subjects = dados.get("subjects", [])

    if not subjects:
        return assuntos

    for subject in subjects:

        subject_limpo = normalizar_subject(subject)

        if not subject_limpo:
            continue

        if not eh_genero_ou_tema(subject_limpo):
            continue

        # Evita duplicatas ignorando maiúsculas/minúsculas
        ja_existe = any(
            subject_limpo.lower() == existente.lower()
            for existente in assuntos
        )

        if ja_existe:
            continue

        assuntos.append(subject_limpo)

    return assuntos


# ============================================================
# BUSCAR POR ISBN
# ============================================================

def buscar_por_isbn(isbn):

    isbn = limpar_isbn(isbn)

    if not isbn:
        return None

    try:

        resposta = requests.get(
            URL_ISBN.format(isbn),
            headers=HEADERS,
            timeout=15
        )

        if resposta.status_code == 404:
            return None

        if resposta.status_code != 200:
            return None

        return resposta.json()

    except requests.RequestException:

        return None


# ============================================================
# BUSCAR POR TÍTULO + AUTOR
# ============================================================

def buscar_por_titulo_autor(titulo, autor):

    titulo = limpar_titulo(titulo)
    autor = limpar_autor(autor)

    if not titulo:
        return None

    try:

        parametros = {
            "title": titulo,
            "author": autor,
            "limit": 5
        }

        resposta = requests.get(
            URL_SEARCH,
            params=parametros,
            headers=HEADERS,
            timeout=15
        )

        if resposta.status_code != 200:
            return None

        dados = resposta.json()

        documentos = dados.get("docs", [])

        if not documentos:
            return None

        return documentos[0]

    except requests.RequestException:

        return None


# ============================================================
# BUSCAR PELA OBRA
# ============================================================

def buscar_por_work(work_key):

    if not work_key:
        return None

    try:

        work_id = work_key.split("/")[-1]

        resposta = requests.get(
            URL_WORK.format(work_id),
            headers=HEADERS,
            timeout=15
        )

        if resposta.status_code != 200:
            return None

        return resposta.json()

    except requests.RequestException:

        return None


# ============================================================
# OBTER GÊNEROS
# ============================================================

def obter_generos(titulo, autor, isbn):

    # --------------------------------------------------------
    # 1. ISBN
    # --------------------------------------------------------

    dados_isbn = buscar_por_isbn(isbn)

    if dados_isbn:

        assuntos = extrair_assuntos(dados_isbn)

        if assuntos:
            return assuntos[:8]


        # ----------------------------------------------------
        # 2. OBRA RELACIONADA AO ISBN
        # ----------------------------------------------------

        works = dados_isbn.get("works", [])

        if works:

            primeira_obra = works[0]

            if isinstance(primeira_obra, dict):

                work_key = primeira_obra.get("key")

            else:

                work_key = primeira_obra


            dados_work = buscar_por_work(work_key)

            if dados_work:

                assuntos = extrair_assuntos(
                    dados_work
                )

                if assuntos:
                    return assuntos[:8]


    # --------------------------------------------------------
    # 3. FALLBACK: TÍTULO + AUTOR
    # --------------------------------------------------------

    dados_busca = buscar_por_titulo_autor(
        titulo,
        autor
    )

    if dados_busca:

        assuntos = extrair_assuntos(
            dados_busca
        )

        if assuntos:
            return assuntos[:8]


    # --------------------------------------------------------
    # 4. NENHUM RESULTADO
    # --------------------------------------------------------

    return []


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # Validar número do lote
    # --------------------------------------------------------

    if LOTE < 1 or LOTE > TOTAL_LOTES:

        print(
            f"ERRO: LOTE deve estar entre 1 e "
            f"{TOTAL_LOTES}."
        )

        return


    print("Lendo books_corrigido.csv...")


    # --------------------------------------------------------
    # Ler arquivo
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            ARQUIVO_ENTRADA,
            encoding="utf-8-sig"
        )

    except FileNotFoundError:

        print(
            f"\nERRO: O arquivo '{ARQUIVO_ENTRADA}' "
            "não foi encontrado."
        )

        return


    # --------------------------------------------------------
    # Limpar nomes das colunas
    # --------------------------------------------------------

    df.columns = df.columns.str.strip()


    total_registros = len(df)


    print(
        f"Total de registros encontrados: "
        f"{total_registros}"
    )


    # ========================================================
    # CALCULAR LOTE
    # ========================================================

    tamanho_base = total_registros // TOTAL_LOTES
    resto = total_registros % TOTAL_LOTES


    # Os primeiros lotes recebem um registro adicional
    if LOTE <= resto:

        inicio = (LOTE - 1) * (tamanho_base + 1)
        fim = inicio + tamanho_base + 1

    else:

        inicio = (
            resto * (tamanho_base + 1)
            + (LOTE - resto - 1) * tamanho_base
        )

        fim = inicio + tamanho_base


    df_lote = df.iloc[inicio:fim].copy()


    # --------------------------------------------------------
    # Nome do arquivo de saída
    # --------------------------------------------------------

    arquivo_saida = (
        f"base_com_generos_lote_{LOTE}.csv"
    )


    print(
        f"\nLote atual: {LOTE}/{TOTAL_LOTES}"
    )

    print(
        f"Índice inicial: {inicio}"
    )

    print(
        f"Índice final: {fim - 1}"
    )

    print(
        f"Quantidade de registros neste lote: "
        f"{len(df_lote)}"
    )


    # --------------------------------------------------------
    # Criar coluna genero
    # --------------------------------------------------------

    df_lote["genero"] = ""


    print(
        "\nIniciando busca de gêneros/temas...\n"
    )


    encontrados = 0
    desconhecidos = 0


    # ========================================================
    # PROCESSAMENTO
    # ========================================================

    for indice, linha in tqdm(
        df_lote.iterrows(),
        total=len(df_lote)
    ):

        titulo = linha["title"]
        autor = linha["authors"]
        isbn = linha["isbn"]


        generos = obter_generos(
            titulo,
            autor,
            isbn
        )


        # ----------------------------------------------------
        # Salvar resultado
        # ----------------------------------------------------

        if generos:

            df_lote.at[
                indice,
                "genero"
            ] = ", ".join(generos)

            encontrados += 1

        else:

            df_lote.at[
                indice,
                "genero"
            ] = "Desconhecido"

            desconhecidos += 1


        # ----------------------------------------------------
        # Respeitar limite da API
        # ----------------------------------------------------

        time.sleep(1)


    # ========================================================
    # SALVAR LOTE
    # ========================================================

    df_lote.to_csv(
        arquivo_saida,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # RESULTADO
    # ========================================================

    print("\n")
    print("=" * 60)
    print("LOTE CONCLUÍDO!")
    print("=" * 60)

    print(
        f"Lote: {LOTE}/{TOTAL_LOTES}"
    )

    print(
        f"Registros processados: {len(df_lote)}"
    )

    print(
        f"Encontrados: {encontrados}"
    )

    print(
        f"Desconhecidos: {desconhecidos}"
    )

    print(
        f"Arquivo criado: {arquivo_saida}"
    )

    print("=" * 60)


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":
    main()