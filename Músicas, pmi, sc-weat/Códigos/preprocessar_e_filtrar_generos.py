from pathlib import Path
import pandas as pd
import unicodedata
import re
import csv

# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA = Path(r"D:\Downloads")

POSSIVEIS_ARQUIVOS = [
    PASTA / "merged_df.txt",
    PASTA / "merged_df",
    PASTA / "merged_df.csv",
]

ARQUIVO_ENTRADA = None

for arquivo in POSSIVEIS_ARQUIVOS:
    if arquivo.exists():
        ARQUIVO_ENTRADA = arquivo
        break

if ARQUIVO_ENTRADA is None:
    raise FileNotFoundError(
        "Não encontrei merged_df.txt, merged_df ou merged_df.csv em D:\\Downloads"
    )

CHUNKSIZE = 10000

ARQUIVO_SAIDA_LIMPO = PASTA / "merged_df_minusculo_sem_duplicadas.csv"
ARQUIVO_SAIDA_DUPLICADAS = PASTA / "merged_df_duplicadas_removidas.csv"
ARQUIVO_RELATORIO = PASTA / "relatorio_preprocessamento_e_filtragem.csv"

GENEROS_FILTRAR = {
    "pagode": "pagode",
    "sertanejo": "sertanejo",
    "forro": "forró",
    "funk": "funk",
    "mpb": "mpb",
}

ARQUIVOS_SAIDA_GENEROS = {
    "pagode": PASTA / "pagode_tudo_um_csv_com_letras.csv",
    "sertanejo": PASTA / "sertanejo_tudo_um_csv_com_letras.csv",
    "forro": PASTA / "forro_tudo_um_csv_com_letras.csv",
    "funk": PASTA / "funk_tudo_um_csv_com_letras.csv",
    "mpb": PASTA / "mpb_tudo_um_csv_com_letras.csv",
}

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def normalizar_texto(valor):
    """
    Normalização geral:
    - minúsculas;
    - remove acentos;
    - remove espaços repetidos.
    Usada para comparação, busca de colunas e chaves.
    """
    if pd.isna(valor):
        return ""

    texto = str(valor).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"\s+", " ", texto)

    return texto


def normalizar_nome_coluna(coluna):
    """
    Normaliza nome de coluna para encontrar colunas mesmo com:
    - acento;
    - maiúsculas/minúsculas;
    - espaços.
    """
    return normalizar_texto(coluna).replace(" ", "")


def achar_coluna(df, nome_desejado):
    alvo = normalizar_nome_coluna(nome_desejado)

    for coluna in df.columns:
        if normalizar_nome_coluna(coluna) == alvo:
            return coluna

    raise ValueError(
        f"Não encontrei a coluna '{nome_desejado}'. "
        f"Colunas encontradas: {list(df.columns)}"
    )


def para_minusculo(valor):
    """
    Passa para minúsculas preservando acentos.
    Exemplo: 'Forró' vira 'forró', não 'forro'.
    """
    if pd.isna(valor):
        return ""

    texto = str(valor).strip().lower()
    texto = re.sub(r"\s+", " ", texto)

    return texto


def normalizar_para_chave(valor):
    """
    Normalização usada para detectar duplicatas.
    Remove acentos, passa para minúsculas e remove espaços extras.
    """
    return normalizar_texto(valor)


def detectar_encoding(caminho):
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    for enc in encodings:
        try:
            pd.read_csv(
                caminho,
                sep=",",
                encoding=enc,
                nrows=20,
                dtype=str,
                engine="python"
            )
            return enc
        except UnicodeDecodeError:
            continue
        except Exception:
            return enc

    return "latin1"


def separar_generos(genero):
    """
    Separa gêneros quando há algo como:
    'pagode, samba', 'forró / sertanejo', 'funk; pop'.
    """
    genero_norm = normalizar_texto(genero)

    partes = re.split(r"[,;/|]+", genero_norm)
    partes = [p.strip() for p in partes if p.strip()]

    return partes


def contem_genero(genero, genero_alvo):
    """
    Verifica se o campo de gênero contém o gênero-alvo como termo separado.
    Evita pegar correspondências parciais indevidas.
    """
    genero_norm = normalizar_texto(genero)
    genero_alvo_norm = normalizar_texto(genero_alvo)

    return bool(
        re.search(
            rf"(^|[^a-z]){re.escape(genero_alvo_norm)}([^a-z]|$)",
            genero_norm
        )
    )


def classificar_genero(genero, genero_alvo):
    """
    Classifica a música como:
    - genero_vazio;
    - nao_genero;
    - genero_puro;
    - genero_misturado.
    """
    partes = separar_generos(genero)
    genero_alvo_norm = normalizar_texto(genero_alvo)

    if not partes:
        return "genero_vazio"

    tem_genero = any(
        re.search(
            rf"(^|[^a-z]){re.escape(genero_alvo_norm)}([^a-z]|$)",
            parte
        )
        for parte in partes
    )

    if not tem_genero:
        return f"nao_{genero_alvo_norm}"

    outros_generos = [
        parte for parte in partes
        if not re.search(
            rf"(^|[^a-z]){re.escape(genero_alvo_norm)}([^a-z]|$)",
            parte
        )
    ]

    if len(outros_generos) == 0:
        return f"{genero_alvo_norm}_puro"

    return f"{genero_alvo_norm}_misturado"


def contar_palavras(texto):
    if pd.isna(texto):
        return 0

    texto = str(texto).strip()

    if not texto:
        return 0

    return len(re.findall(r"\b\w+\b", texto, flags=re.UNICODE))


def apagar_saidas_antigas():
    arquivos = [
        ARQUIVO_SAIDA_LIMPO,
        ARQUIVO_SAIDA_DUPLICADAS,
        ARQUIVO_RELATORIO,
        *ARQUIVOS_SAIDA_GENEROS.values(),
    ]

    for arquivo in arquivos:
        if arquivo.exists():
            arquivo.unlink()


# ============================================================
# EXECUÇÃO
# ============================================================

encoding = detectar_encoding(ARQUIVO_ENTRADA)

print("=" * 80)
print("PREPROCESSAMENTO + FILTRAGEM POR GÊNERO")
print("=" * 80)
print(f"Arquivo de entrada: {ARQUIVO_ENTRADA}")
print(f"Encoding usado: {encoding}")
print(f"Arquivo limpo sem duplicadas: {ARQUIVO_SAIDA_LIMPO}")
print(f"Arquivo com duplicadas removidas: {ARQUIVO_SAIDA_DUPLICADAS}")
print("Gêneros filtrados: pagode, sertanejo, forró, funk e mpb")
print("=" * 80)

apagar_saidas_antigas()

reader = pd.read_csv(
    ARQUIVO_ENTRADA,
    sep=",",
    encoding=encoding,
    chunksize=CHUNKSIZE,
    dtype=str,
    engine="python",
    on_bad_lines="warn"
)

chaves_vistas = set()

primeira_escrita_limpo = True
primeira_escrita_duplicadas = True

primeira_escrita_genero = {
    genero: True for genero in GENEROS_FILTRAR
}

total_lidas = 0
total_mantidas = 0
total_duplicadas = 0

totais_generos = {
    genero: {
        "total": 0,
        "puro": 0,
        "misturado": 0
    }
    for genero in GENEROS_FILTRAR
}

for chunk in reader:
    chunk = chunk.copy()

    coluna_musica = achar_coluna(chunk, "Nome da Música")
    coluna_artista = achar_coluna(chunk, "Artista")
    coluna_genero = achar_coluna(chunk, "Gênero Musical")
    coluna_letra = achar_coluna(chunk, "Letra da Música")

    inicio_linha_original = total_lidas + 2

    if "linha_original_aproximada" not in chunk.columns:
        chunk.insert(
            0,
            "linha_original_aproximada",
            range(inicio_linha_original, inicio_linha_original + len(chunk))
        )
    else:
        colunas = ["linha_original_aproximada"] + [
            c for c in chunk.columns if c != "linha_original_aproximada"
        ]
        chunk = chunk[colunas]

    total_lidas += len(chunk)

    # --------------------------------------------------------
    # 1) Passa artista, nome da música e gênero para minúsculas
    # --------------------------------------------------------

    chunk[coluna_musica] = chunk[coluna_musica].apply(para_minusculo)
    chunk[coluna_artista] = chunk[coluna_artista].apply(para_minusculo)
    chunk[coluna_genero] = chunk[coluna_genero].apply(para_minusculo)

    # --------------------------------------------------------
    # 2) Remove duplicadas por artista + música + gênero musical
    # --------------------------------------------------------

    chunk["_chave_duplicata"] = (
        chunk[coluna_artista].apply(normalizar_para_chave)
        + " || "
        + chunk[coluna_musica].apply(normalizar_para_chave)
        + " || "
        + chunk[coluna_genero].apply(normalizar_para_chave)
    )

    manter_indices = []
    duplicada_indices = []

    for idx, chave in chunk["_chave_duplicata"].items():
        if chave in chaves_vistas:
            duplicada_indices.append(idx)
        else:
            chaves_vistas.add(chave)
            manter_indices.append(idx)

    df_limpo = chunk.loc[manter_indices].copy()
    df_duplicadas = chunk.loc[duplicada_indices].copy()

    if "_chave_duplicata" in df_limpo.columns:
        df_limpo = df_limpo.drop(columns=["_chave_duplicata"])

    if not df_duplicadas.empty:
        df_duplicadas = df_duplicadas.rename(
            columns={"_chave_duplicata": "chave_duplicata"}
        )

    total_mantidas += len(df_limpo)
    total_duplicadas += len(df_duplicadas)

    # Salva corpus limpo sem duplicadas
    if not df_limpo.empty:
        df_limpo.to_csv(
            ARQUIVO_SAIDA_LIMPO,
            mode="w" if primeira_escrita_limpo else "a",
            header=primeira_escrita_limpo,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL
        )
        primeira_escrita_limpo = False

    # Salva duplicadas removidas para auditoria
    if not df_duplicadas.empty:
        df_duplicadas.to_csv(
            ARQUIVO_SAIDA_DUPLICADAS,
            mode="w" if primeira_escrita_duplicadas else "a",
            header=primeira_escrita_duplicadas,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL
        )
        primeira_escrita_duplicadas = False

    # --------------------------------------------------------
    # 3) Filtra CSVs de pagode, sertanejo, forró, funk e mpb
    #    usando APENAS o df_limpo, ou seja, sem duplicadas
    # --------------------------------------------------------

    for chave_genero, nome_genero in GENEROS_FILTRAR.items():
        arquivo_saida_genero = ARQUIVOS_SAIDA_GENEROS[chave_genero]

        mascara = df_limpo[coluna_genero].apply(
            lambda x: contem_genero(x, nome_genero)
        )

        df_filtrado = df_limpo.loc[mascara].copy()

        if df_filtrado.empty:
            continue

        df_filtrado["genero_filtrado"] = nome_genero
        df_filtrado["genero_normalizado"] = df_filtrado[coluna_genero].apply(normalizar_texto)

        df_filtrado["classificacao_genero_filtrado"] = df_filtrado[coluna_genero].apply(
            lambda x: classificar_genero(x, nome_genero)
        )

        df_filtrado["qtd_generos_detectados"] = df_filtrado[coluna_genero].apply(
            lambda x: len(separar_generos(x))
        )

        df_filtrado["nome_musica_normalizado"] = df_filtrado[coluna_musica].apply(normalizar_texto)
        df_filtrado["artista_normalizado"] = df_filtrado[coluna_artista].apply(normalizar_texto)

        df_filtrado["qtd_caracteres_letra"] = (
            df_filtrado[coluna_letra]
            .fillna("")
            .astype(str)
            .str.len()
        )

        df_filtrado["qtd_palavras_letra"] = df_filtrado[coluna_letra].apply(contar_palavras)

        df_filtrado["status_letra_estimado"] = df_filtrado["qtd_palavras_letra"].apply(
            lambda n: "provavel_completa" if n >= 80 else "provavel_incompleta_ou_trecho"
        )

        df_filtrado["chave_musica_artista"] = (
            df_filtrado["nome_musica_normalizado"]
            + " || "
            + df_filtrado["artista_normalizado"]
        )

        genero_norm = normalizar_texto(nome_genero)

        totais_generos[chave_genero]["total"] += len(df_filtrado)

        totais_generos[chave_genero]["puro"] += (
            df_filtrado["classificacao_genero_filtrado"] == f"{genero_norm}_puro"
        ).sum()

        totais_generos[chave_genero]["misturado"] += (
            df_filtrado["classificacao_genero_filtrado"] == f"{genero_norm}_misturado"
        ).sum()

        df_filtrado.to_csv(
            arquivo_saida_genero,
            mode="w" if primeira_escrita_genero[chave_genero] else "a",
            header=primeira_escrita_genero[chave_genero],
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL
        )

        primeira_escrita_genero[chave_genero] = False

    resumo_generos = " | ".join(
        f"{genero}: {totais_generos[genero]['total']:,}".replace(",", ".")
        for genero in GENEROS_FILTRAR
    )

    print(
        f"Lidas: {total_lidas:,} | "
        f"Mantidas: {total_mantidas:,} | "
        f"Duplicadas: {total_duplicadas:,} | "
        f"{resumo_generos}"
        .replace(",", ".")
    )

# ============================================================
# RELATÓRIO FINAL
# ============================================================

linhas_relatorio = []

linhas_relatorio.append({
    "item": "total_linhas_lidas",
    "valor": total_lidas
})

linhas_relatorio.append({
    "item": "total_linhas_mantidas_sem_duplicadas",
    "valor": total_mantidas
})

linhas_relatorio.append({
    "item": "total_duplicadas_removidas",
    "valor": total_duplicadas
})

for chave_genero, nome_genero in GENEROS_FILTRAR.items():
    linhas_relatorio.append({
        "item": f"{nome_genero}_total_filtrado",
        "valor": totais_generos[chave_genero]["total"]
    })

    linhas_relatorio.append({
        "item": f"{nome_genero}_puro",
        "valor": totais_generos[chave_genero]["puro"]
    })

    linhas_relatorio.append({
        "item": f"{nome_genero}_misturado",
        "valor": totais_generos[chave_genero]["misturado"]
    })

df_relatorio = pd.DataFrame(linhas_relatorio)

df_relatorio.to_csv(
    ARQUIVO_RELATORIO,
    index=False,
    encoding="utf-8-sig",
    quoting=csv.QUOTE_ALL
)

print("\n" + "=" * 80)
print("FINALIZADO")
print("=" * 80)
print(f"Total de linhas lidas: {total_lidas}")
print(f"Total de linhas mantidas sem duplicadas: {total_mantidas}")
print(f"Total de duplicadas removidas: {total_duplicadas}")

print("\nArquivos gerados:")
print(f"- Corpus limpo sem duplicadas: {ARQUIVO_SAIDA_LIMPO}")
print(f"- Duplicadas removidas: {ARQUIVO_SAIDA_DUPLICADAS}")
print(f"- Relatório: {ARQUIVO_RELATORIO}")

for chave_genero, nome_genero in GENEROS_FILTRAR.items():
    print("\n" + "-" * 50)
    print(f"Gênero: {nome_genero}")
    print(f"Total filtrado: {totais_generos[chave_genero]['total']}")
    print(f"{nome_genero} puro: {totais_generos[chave_genero]['puro']}")
    print(f"{nome_genero} misturado com outros gêneros: {totais_generos[chave_genero]['misturado']}")
    print(f"CSV gerado em: {ARQUIVOS_SAIDA_GENEROS[chave_genero]}")