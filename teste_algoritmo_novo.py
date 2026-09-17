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

PESO_SIMILARIDADE = 0.60
PESO_QUALIDADE = 0.15
PESO_POPULARIDADE = 0.05
PESO_GENERO = 0.20


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


df["titulo_normalizado"] = df["title"].apply(
    normalizar_titulo
)


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

X = vectorizer.fit_transform(
    df["texto_modelo"]
)

print(
    f"Características criadas: {X.shape[1]}"
)


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
        df["popularidade"]
        / max_popularidade
    )

else:

    df["popularidade"] = 0


# ============================================================
# 8. ENCONTRAR LIVRO PELO TÍTULO
# ============================================================

def encontrar_livro(titulo_busca):

    busca = titulo_busca.strip().lower()

    resultados = df[
        df["title"]
        .str.lower()
        .str.contains(
            busca,
            na=False
        )
    ]

    return resultados


# ============================================================
# 9. ENCONTRAR LIVROS PELO GÊNERO
# ============================================================

def encontrar_genero(genero_busca):

    busca = genero_busca.strip().lower()

    resultados = df[
        df["genero"]
        .str.lower()
        .str.contains(
            busca,
            na=False
        )
    ]

    return resultados


# ============================================================
# 10. RECOMENDAÇÃO NORMAL PELO TÍTULO
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

    quantidade_vizinhos = min(
        NUMERO_CANDIDATOS,
        len(df)
    )

    distancias, indices = modelo.kneighbors(
        X[indice_livro],
        n_neighbors=quantidade_vizinhos
    )

    indices = indices[0]
    distancias = distancias[0]

    candidatos = []

    titulo_original = df.loc[
        indice_livro,
        "titulo_normalizado"
    ]

    for indice, distancia in zip(
        indices,
        distancias
    ):

        # Não recomendar o próprio livro
        if indice == indice_livro:
            continue

        # Não recomendar título exatamente igual
        if (
            df.loc[
                indice,
                "titulo_normalizado"
            ]
            == titulo_original
        ):
            continue

        similaridade_textual = 1 - distancia

        qualidade = df.loc[
            indice,
            "qualidade"
        ]

        popularidade = df.loc[
            indice,
            "popularidade"
        ]

        score_hibrido = (
            similaridade_textual
            * PESO_SIMILARIDADE

            + qualidade
            * PESO_QUALIDADE

            + popularidade
            * PESO_POPULARIDADE
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

        titulo = df.loc[
            indice,
            "titulo_normalizado"
        ]

        if titulo in titulos_vistos:
            continue

        titulos_vistos.add(titulo)

        recomendacoes.append(candidato)

        if (
            len(recomendacoes)
            >= QUANTIDADE_RECOMENDACOES
        ):
            break


    # ========================================================
    # MOSTRAR RESULTADO
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
            f"\n{posicao}. "
            f"{livro_rec['title']}"
        )

        print(
            f"   Autor: "
            f"{livro_rec['authors']}"
        )

        print(
            f"   Gênero: "
            f"{livro_rec['genero']}"
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
# 11. RECOMENDAÇÃO USANDO TÍTULO + GÊNERO
# ============================================================

def recomendar_por_titulo_e_genero(
    indice_livro,
    genero_busca
):

    livro = df.loc[indice_livro]

    print("\n" + "=" * 70)
    print("LIVRO SELECIONADO")
    print("=" * 70)

    print(f"Título : {livro['title']}")
    print(f"Autor  : {livro['authors']}")
    print(f"Gênero informado: {genero_busca}")

    # --------------------------------------------------------
    # Encontrar livros semelhantes ao título
    # --------------------------------------------------------

    quantidade_vizinhos = min(
        NUMERO_CANDIDATOS,
        len(df)
    )

    distancias, indices = modelo.kneighbors(
        X[indice_livro],
        n_neighbors=quantidade_vizinhos
    )

    indices = indices[0]
    distancias = distancias[0]

    candidatos = []

    titulo_original = df.loc[
        indice_livro,
        "titulo_normalizado"
    ]

    genero_busca_normalizado = (
        genero_busca.strip().lower()
    )

    for indice, distancia in zip(
        indices,
        distancias
    ):

        # Não recomendar o próprio livro
        if indice == indice_livro:
            continue

        # Não recomendar título exatamente igual
        if (
            df.loc[
                indice,
                "titulo_normalizado"
            ]
            == titulo_original
        ):
            continue

        # ----------------------------------------------------
        # Verificar gênero
        # ----------------------------------------------------

        genero_livro = str(
            df.loc[
                indice,
                "genero"
            ]
        ).lower()

        if (
            genero_busca_normalizado
            in genero_livro
        ):

            bonus_genero = 1.0

        else:

            bonus_genero = 0.0


        # ----------------------------------------------------
        # Calcular métricas
        # ----------------------------------------------------

        similaridade_textual = (
            1 - distancia
        )

        qualidade = df.loc[
            indice,
            "qualidade"
        ]

        popularidade = df.loc[
            indice,
            "popularidade"
        ]


        # ----------------------------------------------------
        # Score híbrido
        # ----------------------------------------------------

        score_hibrido = (

            similaridade_textual
            * PESO_SIMILARIDADE

            + qualidade
            * PESO_QUALIDADE

            + popularidade
            * PESO_POPULARIDADE

            + bonus_genero
            * PESO_GENERO
        )


        candidatos.append({
            "indice": indice,
            "similaridade": similaridade_textual,
            "qualidade": qualidade,
            "popularidade": popularidade,
            "bonus_genero": bonus_genero,
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

        titulo = df.loc[
            indice,
            "titulo_normalizado"
        ]

        if titulo in titulos_vistos:
            continue

        titulos_vistos.add(titulo)

        recomendacoes.append(
            candidato
        )

        if (
            len(recomendacoes)
            >= QUANTIDADE_RECOMENDACOES
        ):
            break


    # ========================================================
    # MOSTRAR RESULTADO
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
            f"\n{posicao}. "
            f"{livro_rec['title']}"
        )

        print(
            f"   Autor: "
            f"{livro_rec['authors']}"
        )

        print(
            f"   Gênero: "
            f"{livro_rec['genero']}"
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
            f"   Gênero corresponde: "
            f"{'Sim' if recomendacao['bonus_genero'] > 0 else 'Não'}"
        )

        print(
            f"   Score híbrido: "
            f"{recomendacao['score']:.3f}"
        )

    print("\n" + "=" * 70)


# ============================================================
# 12. RECOMENDAÇÃO APENAS POR GÊNERO
# ============================================================

def recomendar_por_genero(
    genero_busca
):

    resultados = encontrar_genero(
        genero_busca
    )

    if resultados.empty:

        print(
            "\nNenhum livro encontrado "
            "com esse gênero."
        )

        return


    # --------------------------------------------------------
    # Criar score por gênero
    # --------------------------------------------------------

    resultados = resultados.copy()

    resultados["score_genero"] = (

        resultados["qualidade"]
        * 0.75

        +

        resultados["popularidade"]
        * 0.25
    )


    resultados = resultados.sort_values(
        "score_genero",
        ascending=False
    )


    # ========================================================
    # MOSTRAR RESULTADO
    # ========================================================

    print("\n" + "=" * 70)
    print("RECOMENDAÇÕES POR GÊNERO")
    print("=" * 70)

    print(
        f"Gênero informado: "
        f"{genero_busca}"
    )


    titulos_vistos = set()

    contador = 0

    for _, livro in resultados.iterrows():

        titulo = livro[
            "titulo_normalizado"
        ]

        if titulo in titulos_vistos:
            continue

        titulos_vistos.add(titulo)

        contador += 1

        print(
            f"\n{contador}. "
            f"{livro['title']}"
        )

        print(
            f"   Autor: "
            f"{livro['authors']}"
        )

        print(
            f"   Gênero: "
            f"{livro['genero']}"
        )

        print(
            f"   Avaliação: "
            f"{livro['average_rating']:.2f}"
        )

        if contador >= QUANTIDADE_RECOMENDACOES:
            break


    print("\n" + "=" * 70)


# ============================================================
# 13. INTERFACE PARA O USUÁRIO
# ============================================================

while True:

    print("\n")

    titulo_usuario = input(
        "Qual livro você leu? "
    ).strip()


    # --------------------------------------------------------
    # Encerrar
    # --------------------------------------------------------

    if titulo_usuario.lower() in [
        "sair",
        "exit",
        "quit"
    ]:

        print("\nSistema encerrado.")
        break


    # --------------------------------------------------------
    # Entrada vazia
    # --------------------------------------------------------

    if not titulo_usuario:

        print(
            "Digite o nome de um livro."
        )

        continue


    # --------------------------------------------------------
    # Procurar título
    # --------------------------------------------------------

    resultados_titulo = encontrar_livro(
        titulo_usuario
    )


    # --------------------------------------------------------
    # Pedir gênero
    # --------------------------------------------------------

    genero_usuario = input(
        "Qual é o gênero desse livro? "
    ).strip()


    # --------------------------------------------------------
    # Encerrar
    # --------------------------------------------------------

    if genero_usuario.lower() in [
        "sair",
        "exit",
        "quit"
    ]:

        print("\nSistema encerrado.")
        break


    # --------------------------------------------------------
    # Entrada vazia
    # --------------------------------------------------------

    if not genero_usuario:

        print(
            "Digite o gênero do livro."
        )

        continue


    # --------------------------------------------------------
    # Procurar gênero
    # --------------------------------------------------------

    resultados_genero = encontrar_genero(
        genero_usuario
    )


    # ========================================================
    # CASO 1:
    # TÍTULO E GÊNERO ENCONTRADOS
    # ========================================================

    if (
        not resultados_titulo.empty
        and not resultados_genero.empty
    ):

        # ----------------------------------------------------
        # Apenas um resultado para o título
        # ----------------------------------------------------

        if len(resultados_titulo) == 1:

            indice_escolhido = (
                resultados_titulo.index[0]
            )

            recomendar_por_titulo_e_genero(
                indice_escolhido,
                genero_usuario
            )

            continue


        # ----------------------------------------------------
        # Vários resultados para o título
        # ----------------------------------------------------

        print(
            f"\nForam encontrados "
            f"{len(resultados_titulo)} livros:"
        )

        opcoes = resultados_titulo.head(10)


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

        while True:

            escolha = input(
                "\nDigite o número do livro: "
            ).strip()


            if escolha.lower() in [
                "sair",
                "exit",
                "quit"
            ]:

                print(
                    "\nSistema encerrado."
                )

                raise SystemExit


            if escolha == "0":
                break


            if escolha.isdigit():

                numero = int(escolha)

                if (
                    1 <= numero
                    <= len(opcoes)
                ):

                    indice_escolhido = (
                        opcoes.index[
                            numero - 1
                        ]
                    )

                    recomendar_por_titulo_e_genero(
                        indice_escolhido,
                        genero_usuario
                    )

                    break


            print(
                "Opção inválida. "
                "Digite um número da lista."
            )


        continue


    # ========================================================
    # CASO 2:
    # TÍTULO ENCONTRADO
    # GÊNERO NÃO ENCONTRADO
    # ========================================================

    if not resultados_titulo.empty:

        print(
            "\nO título foi encontrado, "
            "mas o gênero informado "
            "não foi encontrado na base."
        )

        print(
            "Vamos recomendar usando "
            "apenas o título."
        )


        # ----------------------------------------------------
        # Apenas um resultado
        # ----------------------------------------------------

        if len(resultados_titulo) == 1:

            indice_escolhido = (
                resultados_titulo.index[0]
            )

            recomendar_livros_hibrido(
                indice_escolhido
            )

            continue


        # ----------------------------------------------------
        # Vários resultados
        # ----------------------------------------------------

        print(
            f"\nForam encontrados "
            f"{len(resultados_titulo)} livros:"
        )

        opcoes = resultados_titulo.head(10)


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


        while True:

            escolha = input(
                "\nDigite o número do livro: "
            ).strip()


            if escolha.lower() in [
                "sair",
                "exit",
                "quit"
            ]:

                print(
                    "\nSistema encerrado."
                )

                raise SystemExit


            if escolha == "0":
                break


            if escolha.isdigit():

                numero = int(escolha)

                if (
                    1 <= numero
                    <= len(opcoes)
                ):

                    indice_escolhido = (
                        opcoes.index[
                            numero - 1
                        ]
                    )

                    recomendar_livros_hibrido(
                        indice_escolhido
                    )

                    break


            print(
                "Opção inválida. "
                "Digite um número da lista."
            )


        continue


    # ========================================================
    # CASO 3:
    # TÍTULO NÃO ENCONTRADO
    # GÊNERO ENCONTRADO
    # ========================================================

    if (
        resultados_titulo.empty
        and not resultados_genero.empty
    ):

        print(
            "\nO título não foi encontrado "
            "na base."
        )

        print(
            f"Mas encontramos livros "
            f"do gênero '{genero_usuario}'."
        )

        recomendar_por_genero(
            genero_usuario
        )

        continue


    # ========================================================
    # CASO 4:
    # NEM TÍTULO NEM GÊNERO ENCONTRADOS
    # ========================================================

    print(
        "\nNão encontramos o título "
        "nem o gênero informado."
    )