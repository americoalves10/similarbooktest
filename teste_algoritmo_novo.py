import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO = "books_enriquecidos_final_corrigido.csv"

QUANTIDADE_RECOMENDACOES = 10
NUMERO_CANDIDATOS = 100

PESO_SIMILARIDADE = 0.70
PESO_QUALIDADE = 0.20
PESO_POPULARIDADE = 0.10


# ============================================================
# 1. CARREGAR BASE
# ============================================================

print("=" * 70)
print("SISTEMA DE RECOMENDAÇÃO DE LIVROS")
print("=" * 70)

print("\nCarregando base...")

df = pd.read_csv(ARQUIVO)

print(f"Base carregada: {len(df)} livros.")


# ============================================================
# 2. PREPARAÇÃO DOS DADOS
# ============================================================

df["title"] = df["title"].fillna("").astype(str)
df["authors"] = df["authors"].fillna("").astype(str)
df["genero"] = df["genero"].fillna("").astype(str)

df["average_rating"] = pd.to_numeric(
    df["average_rating"],
    errors="coerce"
).fillna(0)

df["ratings_count"] = pd.to_numeric(
    df["ratings_count"],
    errors="coerce"
).fillna(0)


# ============================================================
# 3. NORMALIZAR TÍTULO
# ============================================================

def normalizar_titulo(titulo):

    titulo = str(titulo).lower()

    # Remove conteúdo entre parênteses
    titulo = re.sub(r"\([^)]*\)", "", titulo)

    # Remove caracteres especiais
    titulo = re.sub(r"[^a-z0-9\s]", " ", titulo)

    # Remove espaços duplicados
    titulo = re.sub(r"\s+", " ", titulo)

    return titulo.strip()


df["titulo_normalizado"] = df["title"].apply(normalizar_titulo)


# ============================================================
# 4. CRIAR REPRESENTAÇÃO TEXTUAL
# ============================================================

# Repetimos o gênero para dar mais peso a essa informação
df["texto_modelo"] = (
    df["title"] + " "
    + df["authors"] + " "
    + df["genero"] + " "
    + df["genero"]
)


print("\nCriando representação TF-IDF...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

X = vectorizer.fit_transform(df["texto_modelo"])

print(f"Características criadas: {X.shape[1]}")


# ============================================================
# 5. MODELO DE VIZINHOS
# ============================================================

modelo = NearestNeighbors(
    n_neighbors=NUMERO_CANDIDATOS,
    metric="cosine"
)

modelo.fit(X)

print("Modelo de recomendação pronto.")


# ============================================================
# 6. NORMALIZAÇÃO DE QUALIDADE
# ============================================================

df["qualidade"] = (
    df["average_rating"] / 5
)


# ============================================================
# 7. NORMALIZAÇÃO DE POPULARIDADE
# ============================================================

df["popularidade"] = np.log1p(
    df["ratings_count"]
)

max_popularidade = df["popularidade"].max()

if max_popularidade > 0:

    df["popularidade"] = (
        df["popularidade"] / max_popularidade
    )

else:

    df["popularidade"] = 0


# ============================================================
# 8. ENCONTRAR LIVRO
# ============================================================

def encontrar_livro(titulo_busca):

    busca = titulo_busca.strip().lower()

    resultados = df[
        df["title"]
        .str.lower()
        .str.contains(busca, na=False)
    ]

    return resultados


# ============================================================
# 9. RECOMENDAÇÃO
# ============================================================

def recomendar_livros_hibrido(indice_livro):

    livro = df.loc[indice_livro]

    print("\n" + "=" * 70)
    print("LIVRO SELECIONADO")
    print("=" * 70)

    print(f"Título : {livro['title']}")
    print(f"Autor  : {livro['authors']}")
    print(f"Gênero : {livro['genero']}")

    # --------------------------------------------------------
    # Encontrar livros semelhantes
    # --------------------------------------------------------

    distancias, indices = modelo.kneighbors(
        X[indice_livro],
        n_neighbors=NUMERO_CANDIDATOS
    )

    indices = indices[0]
    distancias = distancias[0]

    candidatos = []

    titulo_original = df.loc[indice_livro, "titulo_normalizado"]

    for indice, distancia in zip(indices, distancias):

        # Não recomendar o próprio livro
        if indice == indice_livro:
            continue

        # Não recomendar título exatamente igual
        if df.loc[indice, "titulo_normalizado"] == titulo_original:
            continue

        similaridade_textual = 1 - distancia

        qualidade = df.loc[indice, "qualidade"]

        popularidade = df.loc[indice, "popularidade"]

        score_hibrido = (
            similaridade_textual * PESO_SIMILARIDADE
            + qualidade * PESO_QUALIDADE
            + popularidade * PESO_POPULARIDADE
        )

        candidatos.append({
            "indice": indice,
            "similaridade": similaridade_textual,
            "qualidade": qualidade,
            "popularidade": popularidade,
            "score": score_hibrido
        })


    # --------------------------------------------------------
    # Ordenar recomendações
    # --------------------------------------------------------

    candidatos = sorted(
        candidatos,
        key=lambda x: x["score"],
        reverse=True
    )


    # --------------------------------------------------------
    # Remover títulos duplicados
    # --------------------------------------------------------

    recomendacoes = []

    titulos_vistos = set()

    for candidato in candidatos:

        indice = candidato["indice"]

        titulo = df.loc[indice, "titulo_normalizado"]

        if titulo in titulos_vistos:
            continue

        titulos_vistos.add(titulo)

        recomendacoes.append(candidato)

        if len(recomendacoes) >= QUANTIDADE_RECOMENDACOES:
            break


    # ========================================================
    # 10. MOSTRAR RESULTADO
    # ========================================================

    print("\n" + "=" * 70)
    print("LIVROS RECOMENDADOS")
    print("=" * 70)

    for posicao, recomendacao in enumerate(
        recomendacoes,
        start=1
    ):

        indice = recomendacao["indice"]

        livro_rec = df.loc[indice]

        print(
            f"\n{posicao}. {livro_rec['title']}"
        )

        print(
            f"   Autor: {livro_rec['authors']}"
        )

        print(
            f"   Gênero: {livro_rec['genero']}"
        )

        print(
            f"   Avaliação: "
            f"{livro_rec['average_rating']:.2f}"
        )

        print(
            f"   Similaridade: "
            f"{recomendacao['similaridade']:.3f}"
        )

        print(
            f"   Score híbrido: "
            f"{recomendacao['score']:.3f}"
        )

    print("\n" + "=" * 70)


# ============================================================
# 11. INTERFACE PARA O USUÁRIO
# ============================================================

while True:

    print("\n")
    titulo_usuario = input(
        "Qual livro você leu? "
    ).strip()

    # Encerrar
    if titulo_usuario.lower() in [
        "sair",
        "exit",
        "quit"
    ]:

        print("\nSistema encerrado.")
        break


    # Entrada vazia
    if not titulo_usuario:

        print("Digite o nome de um livro.")

        continue


    # --------------------------------------------------------
    # Procurar livro
    # --------------------------------------------------------

    resultados = encontrar_livro(
        titulo_usuario
    )


    if resultados.empty:

        print(
            "\nNenhum livro encontrado "
            "com esse título."
        )

        continue


    # --------------------------------------------------------
    # Apenas um resultado
    # --------------------------------------------------------

    if len(resultados) == 1:

        indice_escolhido = resultados.index[0]

        recomendar_livros_hibrido(
            indice_escolhido
        )

        continue


    # --------------------------------------------------------
    # Vários resultados
    # --------------------------------------------------------

    print(
        f"\nForam encontrados "
        f"{len(resultados)} livros:"
    )

    opcoes = resultados.head(10)

    for numero, (indice, livro) in enumerate(
        opcoes.iterrows(),
        start=1
    ):

        print(
            f"{numero}. "
            f"{livro['title']} "
            f"- {livro['authors']}"
        )


    print("\nDigite 0 para voltar.")
    print("Digite 'sair' para encerrar.")


    while True:

        escolha = input(
            "\nDigite o número do livro: "
        ).strip()


        # Sair
        if escolha.lower() in [
            "sair",
            "exit",
            "quit"
        ]:

            print("\nSistema encerrado.")
            raise SystemExit


        # Voltar
        if escolha == "0":

            break


        # Escolha numérica
        if escolha.isdigit():

            numero = int(escolha)

            if 1 <= numero <= len(opcoes):

                indice_escolhido = opcoes.index[
                    numero - 1
                ]

                recomendar_livros_hibrido(
                    indice_escolhido
                )

                break


        print(
            "Opção inválida. "
            "Digite um número da lista."
        )