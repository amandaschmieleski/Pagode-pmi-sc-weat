# -*- coding: utf-8 -*-
"""
WEAT + SC-WEAT para corpus de músicas, usando listas ALVOS/ATRIBUTOS
reaproveitadas do código de PMI.

Este script gera DOIS formatos de gráfico:

1) SC-WEAT estilo Chen et al. (2025)
   - Uma barra por categoria: Agradável, Desagradável, Aparência,
     Inteligência, Força, Fraqueza.
   - A = Masculino; B = Feminino.
   - d > 0: viés masculino.
   - d < 0: viés feminino.
   - Parecido com a figura do artigo Chen: eixo X = Target Set; eixo Y = Effect Size.

2) SC-WEAT estilo TCC/primeiro print
   - Pares no eixo X: Agradável/Desagradável, Aparência/Inteligência,
     Força/Fraqueza.
   - Barras lado a lado para Masculino e Feminino.
   - O gráfico mostra MAGNITUDE DO EFEITO positiva, para ficar no formato do print.
   - O CSV mantém também o sinal do d de Cohen.

Também gera tabelas:
- tabela SC-WEAT estilo TCC, com SC-WEAT, significância e magnitude do efeito;
- tabela comparativa WEAT/SC-WEAT, no estilo do segundo print;
- auditoria das palavras que entraram/saíram do vocabulário.

Arquivos esperados na mesma pasta:
    bias_weat_scweat_somente_merged_df.py
    pmi_pagode_diagnostico_sem_dedup_v3_4graficos_pmi_atributos_ordenados.py
    merged_df   OU merged_df.csv   OU merged_df.txt

Este script NÃO usa musicas_pagode_corrigido.csv como fallback.
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from gensim.models import Word2Vec
except ImportError as exc:
    raise RuntimeError(
        "gensim não está instalado. Instale com:\n"
        "  pip install gensim\n"
        "Se aparecer erro relacionado ao scipy, atualize também:\n"
        "  pip install -U scipy gensim"
    ) from exc

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

# O script procura o primeiro arquivo existente nessa ordem.
POSSIVEIS_ARQUIVOS_DADOS = [
    "merged_df",
    "merged_df.csv",
    "merged_df.txt",
]

# Se quiser forçar um caminho específico, coloque aqui. Ex.: r"D:\\Downloads\\merged_df"
CAMINHO_DADOS_FIXO = None

# Se deixar None, o script tenta detectar automaticamente.
COLUNA_TEXTO = None
POSSIVEIS_COLUNAS_TEXTO = [
    "Letra da Música",
    "letra da música",
    "letra de música",
    "Letra",
    "letra",
    "lyrics",
    "Lyrics",
    "texto",
    "Texto",
]

CAMINHO_CODIGO_PMI = "COMPLETO_146k_pmi_alfabetico.py"
PREFIXO_SAIDA = "14_junho_bias_weat_scweat_somente_merged_df"

# ============================================================
# PARÂMETROS WORD2VEC
# ============================================================
# Chen et al. usam Word2Vec treinado do zero para medir o viés do próprio corpus.
# O artigo não especifica no texto principal todos os hiperparâmetros, então estes
# são valores práticos e reprodutíveis para corpus em português.

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 1       # mantém termos raros dos alvos/atributos
SG = 1              # 1 = skip-gram; 0 = CBOW
EPOCHS = 50
WORKERS = 4
SEED = 42

PRESERVAR_ACENTOS_NOS_TOKENS = True
REMOVER_STOPWORDS = True

# Permutações para p-valor. A figura do Chen foca no effect size; aqui mantemos
# significância para montar tabelas acadêmicas.
N_PERMUTACOES = 1000
P_VALOR_BICAUDAL = True

# Balanceamento por frequência para comparações simétricas.
BALANCEAR_MASC_FEM = True
BALANCEAR_PARES_ATRIBUTOS = True

ORDEM_CATEGORIAS = [
    "Agradável",
    "Desagradável",
    "Aparência",
    "Inteligência",
    "Força",
    "Fraqueza",
]

LABELS_CHEN = {
    "Agradável": "Pleasant",
    "Desagradável": "Unpleasant",
    "Aparência": "Appearance",
    "Inteligência": "Intelligence",
    "Força": "Strength",
    "Fraqueza": "Weakness",
}

PARES_CLASSICOS = [
    ("Agradável", "Desagradável"),
    ("Aparência", "Inteligência"),
    ("Força", "Fraqueza"),
]

# Pares adicionais para a tabela comparativa. Escolhi os mais úteis para discutir
# estereótipos em letras: valência, corpo/competência, força/fraqueza, corpo/força.
PARES_TABELA_COMPARATIVA = [
    ("Agradável", "Desagradável"),
    ("Aparência", "Inteligência"),
    ("Força", "Fraqueza"),
    ("Aparência", "Força"),
    ("Aparência", "Agradável"),
    ("Desagradável", "Fraqueza"),
    ("Inteligência", "Força"),
]

STOPWORDS_PT_MIN = {
    "a", "à", "ao", "aos", "as", "às", "com", "como", "da", "das", "de",
    "do", "dos", "e", "é", "em", "eu", "foi", "me", "meu", "meus", "minha",
    "minhas", "na", "não", "nas", "no", "nos", "nós", "o", "os", "ou", "pra",
    "para", "por", "que", "se", "sem", "ser", "seu", "seus", "sua", "suas",
    "te", "tem", "ter", "tu", "um", "uma", "você", "vocês", "vou", "tô",
    "ta", "tá",
}

# ============================================================
# LEITURA ROBUSTA DO ARQUIVO
# ============================================================


def localizar_arquivo_dados() -> Path:
    if CAMINHO_DADOS_FIXO:
        p = Path(CAMINHO_DADOS_FIXO)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo fixo não encontrado: {p.resolve()}")
        return p
    for nome in POSSIVEIS_ARQUIVOS_DADOS:
        p = Path(nome)
        if p.exists():
            return p
    raise FileNotFoundError(
        "Não encontrei o merged_df. Coloque na mesma pasta um destes nomes ou configure CAMINHO_DADOS_FIXO:\n"
        + "\n".join(f"- {x}" for x in POSSIVEIS_ARQUIVOS_DADOS)
    )


def detectar_encoding_e_separador(caminho: Path) -> tuple[str, str]:
    encodings = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    amostra_bytes = caminho.read_bytes()[:200_000]
    ultimo_erro = None
    for enc in encodings:
        try:
            texto = amostra_bytes.decode(enc)
            try:
                dialect = csv.Sniffer().sniff(texto, delimiters=",;\t|")
                return enc, dialect.delimiter
            except Exception:
                # fallback simples por contagem
                candidatos = [",", ";", "\t", "|"]
                sep = max(candidatos, key=lambda s: texto.count(s))
                return enc, sep
        except Exception as exc:
            ultimo_erro = exc
    raise RuntimeError(f"Não consegui detectar encoding do arquivo {caminho}: {ultimo_erro}")


def ler_dataframe(caminho: Path) -> pd.DataFrame:
    enc, sep = detectar_encoding_e_separador(caminho)
    print(f"Lendo arquivo: {caminho} | encoding={enc} | sep={repr(sep)}")
    try:
        return pd.read_csv(caminho, encoding=enc, sep=sep, dtype=str, low_memory=False)
    except Exception as exc1:
        print("Leitura padrão falhou. Tentando leitura com engine='python' e on_bad_lines='warn'...")
        try:
            return pd.read_csv(caminho, encoding=enc, sep=sep, dtype=str, engine="python", on_bad_lines="warn")
        except Exception as exc2:
            raise RuntimeError(
                f"Não consegui ler o arquivo {caminho}.\n"
                f"Erro 1: {exc1}\nErro 2: {exc2}"
            )


def normalizar_nome_coluna(nome: str) -> str:
    s = unicodedata.normalize("NFKD", str(nome).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def detectar_coluna_texto(df: pd.DataFrame) -> str:
    if COLUNA_TEXTO and COLUNA_TEXTO in df.columns:
        return COLUNA_TEXTO

    mapa = {normalizar_nome_coluna(c): c for c in df.columns}
    for c in POSSIVEIS_COLUNAS_TEXTO:
        chave = normalizar_nome_coluna(c)
        if chave in mapa:
            return mapa[chave]

    # fallback: escolhe a coluna textual com maior tamanho médio.
    melhores = []
    for c in df.columns:
        serie = df[c].dropna().astype(str)
        if serie.empty:
            continue
        media_len = serie.head(5000).str.len().mean()
        melhores.append((media_len, c))
    melhores.sort(reverse=True)
    if melhores and melhores[0][0] >= 40:
        print(
            "ATENÇÃO: coluna de letra detectada automaticamente como "
            f"'{melhores[0][1]}' (tamanho médio amostral {melhores[0][0]:.1f})."
        )
        return melhores[0][1]

    raise ValueError(
        "Não consegui detectar a coluna de letras. Colunas encontradas:\n"
        + "\n".join(f"- {c}" for c in df.columns)
        + "\n\nAjuste COLUNA_TEXTO no topo do script."
    )

# ============================================================
# NORMALIZAÇÃO E TOKENIZAÇÃO
# ============================================================


def remover_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def normalizar_token(texto: str) -> str:
    texto = str(texto).lower().strip()
    if not PRESERVAR_ACENTOS_NOS_TOKENS:
        texto = remover_acentos(texto)
    texto = re.sub(r"[^a-zà-öø-ÿ0-9]", "", texto)
    return texto


def normalizar_lista_palavras(palavras: Iterable[str]) -> list[str]:
    saida = []
    vistos = set()
    for palavra in palavras:
        norm = normalizar_token(palavra)
        if norm and norm not in vistos:
            vistos.add(norm)
            saida.append(norm)
    return saida


def carregar_stopwords(palavras_interesse: set[str]) -> set[str]:
    if not REMOVER_STOPWORDS:
        return set()
    try:
        from spacy.lang.pt.stop_words import STOP_WORDS
        stop = {normalizar_token(w) for w in STOP_WORDS if normalizar_token(w)}
    except Exception:
        stop = {normalizar_token(w) for w in STOPWORDS_PT_MIN if normalizar_token(w)}
    # Nunca remove alvos/atributos, mesmo que sejam stopwords.
    return {w for w in stop if w not in palavras_interesse}


def tokenizar_texto(texto: str, stopwords: set[str]) -> list[str]:
    texto = str(texto).lower()
    texto = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s]", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    tokens = []
    for tok in texto.split():
        norm = normalizar_token(tok)
        if not norm:
            continue
        if REMOVER_STOPWORDS and norm in stopwords:
            continue
        tokens.append(norm)
    return tokens

# ============================================================
# CARREGAMENTO DAS LISTAS DO PMI
# ============================================================


def carregar_listas_do_codigo_pmi(caminho_codigo: str | Path):
    caminho = Path(caminho_codigo)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo de PMI não encontrado: {caminho.resolve()}\n"
            "Coloque este script na mesma pasta do código de PMI ou ajuste CAMINHO_CODIGO_PMI."
        )
    spec = importlib.util.spec_from_file_location("pmi_base", caminho)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Não consegui importar o arquivo: {caminho}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    if not hasattr(modulo, "ALVOS") or not hasattr(modulo, "ATRIBUTOS"):
        raise AttributeError("O arquivo de PMI precisa ter as variáveis ALVOS e ATRIBUTOS.")
    return modulo.ALVOS, modulo.ATRIBUTOS


def coletar_palavras_interesse(alvos: dict, atributos: dict) -> set[str]:
    palavras = set()
    for lista in alvos.values():
        palavras.update(normalizar_lista_palavras(lista))
    for lista in atributos.values():
        palavras.update(normalizar_lista_palavras(lista))
    return palavras

# ============================================================
# PREPARAÇÃO DOS CONJUNTOS
# ============================================================


def filtrar_vocabulario(palavras: Iterable[str], vocab_freq: dict[str, int]) -> list[str]:
    return [p for p in palavras if p in vocab_freq]


def preparar_conjunto(nome: str, palavras: Iterable[str], vocab_freq: dict[str, int]) -> tuple[list[str], list[dict]]:
    palavras_norm = normalizar_lista_palavras(palavras)
    usadas = filtrar_vocabulario(palavras_norm, vocab_freq)
    auditoria = []
    for p in palavras_norm:
        auditoria.append({
            "conjunto": nome,
            "palavra_norm": p,
            "presente_vocab": p in vocab_freq,
            "frequencia_vocab": vocab_freq.get(p, 0),
        })
    return usadas, auditoria


def balancear_por_frequencia(a: list[str], b: list[str], vocab_freq: dict[str, int]) -> tuple[list[str], list[str]]:
    n = min(len(a), len(b))
    a_ord = sorted(a, key=lambda w: (-vocab_freq.get(w, 0), w))[:n]
    b_ord = sorted(b, key=lambda w: (-vocab_freq.get(w, 0), w))[:n]
    return sorted(a_ord), sorted(b_ord)

# ============================================================
# CÁLCULOS WEAT / SC-WEAT
# ============================================================


def cosine(model: Word2Vec, w1: str, w2: str) -> float:
    return float(model.wv.similarity(w1, w2))


def s_palavra(model: Word2Vec, w: str, A: list[str], B: list[str]) -> float:
    if not A or not B:
        return float("nan")
    media_a = float(np.mean([cosine(model, w, a) for a in A]))
    media_b = float(np.mean([cosine(model, w, b) for b in B]))
    return media_a - media_b


def valores_s(model: Word2Vec, W: list[str], A: list[str], B: list[str]) -> np.ndarray:
    vals = np.array([s_palavra(model, w, A, B) for w in W], dtype=float)
    return vals[~np.isnan(vals)]


def scweat_set(model: Word2Vec, W: list[str], A: list[str], B: list[str]) -> dict:
    vals = valores_s(model, W, A, B)
    if vals.size == 0:
        return {"scweat_soma": np.nan, "scweat_media": np.nan, "std_s": np.nan, "effect_size_d": np.nan}
    media = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
    d = float(media / std) if std and not math.isnan(std) else np.nan
    return {"scweat_soma": float(np.sum(vals)), "scweat_media": media, "std_s": std, "effect_size_d": d}


def permutation_signflip_scweat(vals: np.ndarray, n_permutacoes: int, bicaudal: bool, seed: int) -> tuple[float, float, float]:
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    rng = random.Random(seed)
    observado = float(np.mean(vals))
    perm = []
    for _ in range(n_permutacoes):
        sinais = np.array([1 if rng.random() >= 0.5 else -1 for _ in vals], dtype=float)
        perm.append(float(np.mean(vals * sinais)))
    arr = np.array(perm, dtype=float)
    if bicaudal:
        p = (np.sum(np.abs(arr) >= abs(observado)) + 1) / (arr.size + 1)
    else:
        p = (np.sum(arr >= observado) + 1) / (arr.size + 1)
    return float(p), float(np.mean(arr)), float(np.std(arr, ddof=1))


def precomputar_s(model: Word2Vec, palavras: list[str], A: list[str], B: list[str]) -> dict[str, float]:
    return {w: s_palavra(model, w, A, B) for w in palavras}


def estatistica_S_precomp(svals: dict[str, float], X: list[str], Y: list[str]) -> float:
    return float(sum(svals[x] for x in X) - sum(svals[y] for y in Y))


def effect_size_weat_precomp(svals: dict[str, float], X: list[str], Y: list[str]) -> float:
    sx = np.array([svals[x] for x in X], dtype=float)
    sy = np.array([svals[y] for y in Y], dtype=float)
    todos = np.concatenate([sx, sy])
    desvio = float(np.std(todos, ddof=1))
    if desvio == 0 or math.isnan(desvio):
        return float("nan")
    return float((np.mean(sx) - np.mean(sy)) / desvio)


def permutation_test_weat_precomp(svals: dict[str, float], X: list[str], Y: list[str], n_permutacoes: int, bicaudal: bool, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    observado = estatistica_S_precomp(svals, X, Y)
    I = list(dict.fromkeys(X + Y))
    n_x = len(X)
    if len(I) != len(X) + len(Y):
        raise ValueError("Há interseção entre X e Y depois da normalização.")
    total_combinacoes = math.comb(len(I), n_x)
    permutados = []
    if total_combinacoes <= n_permutacoes:
        for Xi_tuple in itertools.combinations(I, n_x):
            Xi = list(Xi_tuple)
            Xi_set = set(Xi)
            Yi = [w for w in I if w not in Xi_set]
            permutados.append(estatistica_S_precomp(svals, Xi, Yi))
    else:
        vistos = set()
        tentativas = 0
        while len(permutados) < n_permutacoes and tentativas < n_permutacoes * 30:
            tentativas += 1
            Xi = tuple(sorted(rng.sample(I, n_x)))
            if Xi in vistos:
                continue
            vistos.add(Xi)
            Xi_list = list(Xi)
            Xi_set = set(Xi_list)
            Yi_list = [w for w in I if w not in Xi_set]
            permutados.append(estatistica_S_precomp(svals, Xi_list, Yi_list))
    arr = np.array(permutados, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if bicaudal:
        p = (np.sum(np.abs(arr) >= abs(observado)) + 1) / (arr.size + 1)
    else:
        p = (np.sum(arr > observado) + 1) / (arr.size + 1)
    return float(p), float(np.mean(arr)), float(np.std(arr, ddof=1))

# ============================================================
# FORMATAÇÃO / GRÁFICOS / TABELAS
# ============================================================


def estrela(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fmt_num(x: float, casas: int = 2, dec_comma: bool = True) -> str:
    if pd.isna(x):
        return "–"
    s = f"{x:.{casas}f}"
    return s.replace(".", ",") if dec_comma else s


def fmt_num_sig(x: float, p: float, casas: int = 2) -> str:
    return f"{fmt_num(x, casas)}{estrela(p)}"


def grafico_scweat_chen(df: pd.DataFrame, nome_saida: str):
    """Figura estilo Chen: uma barra por categoria, signed effect size."""
    import matplotlib.pyplot as plt

    plot = df.set_index("categoria").reindex(ORDEM_CATEGORIAS).reset_index()
    labels = [LABELS_CHEN.get(c, c) for c in plot["categoria"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(plot))
    ax.bar(x, plot["effect_size_d"], color="#ef8a62")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="-", alpha=0.35)
    ax.set_xlabel("Target Set", fontsize=12)
    ax.set_ylabel("Effect Size", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
    ax.set_title("SC-WEAT effect size", fontsize=13)

    minv = float(np.nanmin(plot["effect_size_d"])) if not plot["effect_size_d"].isna().all() else -0.1
    maxv = float(np.nanmax(plot["effect_size_d"])) if not plot["effect_size_d"].isna().all() else 0.1
    margem = max(0.12, (maxv - minv) * 0.18)
    ax.set_ylim(min(-0.1, minv - margem), max(0.1, maxv + margem))

    plt.tight_layout()
    plt.savefig(nome_saida, dpi=300, bbox_inches="tight")
    plt.close()


def grafico_scweat_pares_tcc(df: pd.DataFrame, nome_saida: str):
    """Figura estilo TCC: pares no eixo X e barras por gênero; usa magnitude positiva."""
    import matplotlib.pyplot as plt

    ordem_labels = [f"{a}/{b}" for a, b in PARES_CLASSICOS]
    plot = df.copy()
    plot["par_label"] = plot["atributo_A"] + "/" + plot["atributo_B"]
    pivot = plot.pivot(index="par_label", columns="alvo", values="magnitude_do_efeito")
    pivot = pivot.reindex(ordem_labels)

    x = np.arange(len(pivot.index))
    largura = 0.36
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.bar(x - largura / 2, pivot.get("Masculino", pd.Series(index=pivot.index, dtype=float)), largura, label="Masculino", color="#7299dc")
    ax.bar(x + largura / 2, pivot.get("Feminino", pd.Series(index=pivot.index, dtype=float)), largura, label="Feminino", color="#e68a8c")
    ax.set_title("Comparativo do SC-WEAT para Gêneros Masculino e Feminino", fontsize=13)
    ax.set_xlabel("Categoria")
    ax.set_ylabel("Magnitude do Efeito")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=50, ha="right")
    ax.legend(title="Gênero")
    ax.set_ylim(0, max(0.1, float(np.nanmax(pivot.values)) * 1.15))
    plt.tight_layout()
    plt.savefig(nome_saida, dpi=300, bbox_inches="tight")
    plt.close()


def tabela_scweat_pares_png(df: pd.DataFrame, nome_saida: str):
    import matplotlib.pyplot as plt

    linhas = []
    for a, b in PARES_CLASSICOS:
        grupo = df[(df["atributo_A"] == a) & (df["atributo_B"] == b)].copy()
        for alvo in ["Feminino", "Masculino"]:
            row = grupo[grupo["alvo"] == alvo]
            if row.empty:
                continue
            r = row.iloc[0]
            linhas.append([
                f"{a}/\n{b}" if alvo == "Feminino" else "",
                alvo,
                fmt_num(r["scweat_media"], 4),
                fmt_num(r["p_value"], 4),
                fmt_num(r["magnitude_do_efeito"], 4),
            ])

    colunas = ["Atributos", "Alvo", "SC-WEAT", "Significância", "Magnitude\ndo Efeito"]
    altura = max(3.2, 0.45 * (len(linhas) + 2))
    fig, ax = plt.subplots(figsize=(10.5, altura))
    ax.axis("off")
    table = ax.table(cellText=linhas, colLabels=colunas, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Tabela - Resultados SC-WEAT para Gêneros Masculino e Feminino", pad=18, fontsize=13)
    fig.text(0.02, 0.02, "*p < 0,10; **p < 0,05", fontsize=9)
    plt.tight_layout()
    plt.savefig(nome_saida, dpi=300, bbox_inches="tight")
    plt.close()


def tabela_comparativa_png(df: pd.DataFrame, nome_saida: str):
    import matplotlib.pyplot as plt

    colunas = ["A", "B", "X", "Y", "corpus", "SC-WEAT X", "SC-WEAT Y", "WEAT\neffect size"]
    linhas = []
    for _, row in df.iterrows():
        linhas.append([
            row["A"], row["B"], row["X"], row["Y"], row["corpus"],
            row["SC-WEAT X fmt"], row["SC-WEAT Y fmt"], row["WEAT effect size fmt"],
        ])

    altura = max(3.4, 0.42 * (len(linhas) + 2))
    fig, ax = plt.subplots(figsize=(13.5, altura))
    ax.axis("off")
    table = ax.table(cellText=linhas, colLabels=colunas, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1, 1.42)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Tabela comparativa WEAT/SC-WEAT - corpus merged_df", fontsize=13, pad=18)
    fig.text(0.02, 0.02, "*p < 0,10; **p < 0,05", fontsize=9)
    plt.tight_layout()
    plt.savefig(nome_saida, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# MAIN
# ============================================================


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    caminho_dados = localizar_arquivo_dados()

    print("Carregando ALVOS e ATRIBUTOS do código de PMI...")
    alvos, atributos = carregar_listas_do_codigo_pmi(CAMINHO_CODIGO_PMI)
    palavras_interesse = coletar_palavras_interesse(alvos, atributos)
    stopwords = carregar_stopwords(palavras_interesse)

    print("Lendo dados...")
    df = ler_dataframe(caminho_dados)
    coluna_texto = detectar_coluna_texto(df)
    print(f"Coluna de texto usada: {coluna_texto}")

    print("Tokenizando letras...")
    sentencas = []
    for texto in df[coluna_texto].fillna("").astype(str):
        toks = tokenizar_texto(texto, stopwords)
        if toks:
            sentencas.append(toks)
    if not sentencas:
        raise ValueError("Nenhuma sentença/token foi gerado. Confira a coluna de letras.")

    print(f"Treinando Word2Vec em {len(sentencas)} documentos...")
    model = Word2Vec(
        sentences=sentencas,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        workers=WORKERS,
        seed=SEED,
        epochs=EPOCHS,
    )
    model.save(f"{PREFIXO_SAIDA}_word2vec.model")
    vocab_freq = {w: int(model.wv.get_vecattr(w, "count")) for w in model.wv.index_to_key}

    print("Preparando conjuntos...")
    auditoria = []
    conjuntos = {}
    for nome, lista in alvos.items():
        usados, audit = preparar_conjunto(nome, lista, vocab_freq)
        conjuntos[nome] = usados
        auditoria.extend({"tipo": "alvo", **x} for x in audit)
    for nome, lista in atributos.items():
        usados, audit = preparar_conjunto(nome, lista, vocab_freq)
        conjuntos[nome] = usados
        auditoria.extend({"tipo": "atributo", **x} for x in audit)

    colisoes = sorted(set(conjuntos.get("Feminino", [])) & set(conjuntos.get("Masculino", [])))
    if colisoes:
        print("ATENÇÃO: removendo colisões entre alvos Feminino/Masculino:", colisoes)
        conjuntos["Feminino"] = [w for w in conjuntos["Feminino"] if w not in colisoes]
        conjuntos["Masculino"] = [w for w in conjuntos["Masculino"] if w not in colisoes]

    pd.DataFrame(auditoria).drop_duplicates().to_csv(f"{PREFIXO_SAIDA}_auditoria_vocabulario.csv", index=False, encoding="utf-8-sig")

    masculino = conjuntos["Masculino"]
    feminino = conjuntos["Feminino"]
    if BALANCEAR_MASC_FEM:
        masculino_bal, feminino_bal = balancear_por_frequencia(masculino, feminino, vocab_freq)
    else:
        masculino_bal, feminino_bal = masculino, feminino

    # --------------------------------------------------------
    # 1) SC-WEAT Chen: W=categoria, A=Masculino, B=Feminino.
    # --------------------------------------------------------
    print("Rodando SC-WEAT estilo Chen...")
    linhas_chen = []
    for categoria in ORDEM_CATEGORIAS:
        W = conjuntos.get(categoria, [])
        vals = valores_s(model, W, masculino_bal, feminino_bal)
        res = scweat_set(model, W, masculino_bal, feminino_bal)
        p, mu_perm, sd_perm = permutation_signflip_scweat(vals, N_PERMUTACOES, P_VALOR_BICAUDAL, SEED)
        linhas_chen.append({
            "categoria": categoria,
            "target_set_label": LABELS_CHEN.get(categoria, categoria),
            "n_W": len(W),
            "n_masculino": len(masculino_bal),
            "n_feminino": len(feminino_bal),
            "SCWEAT_soma": res["scweat_soma"],
            "SCWEAT_media": res["scweat_media"],
            "std_s": res["std_s"],
            "effect_size_d": res["effect_size_d"],
            "magnitude_do_efeito": abs(res["effect_size_d"]) if not pd.isna(res["effect_size_d"]) else np.nan,
            "p_value": p,
            "media_perm": mu_perm,
            "desvio_perm": sd_perm,
            "direcao": "viés masculino" if res["effect_size_d"] > 0 else "viés feminino" if res["effect_size_d"] < 0 else "neutro",
        })
    df_chen = pd.DataFrame(linhas_chen)
    df_chen.to_csv(f"{PREFIXO_SAIDA}_scweat_chen.csv", index=False, encoding="utf-8-sig")
    grafico_scweat_chen(df_chen, f"{PREFIXO_SAIDA}_grafico_scweat_chen.png")

    # --------------------------------------------------------
    # 2) SC-WEAT estilo TCC: W=Feminino/Masculino; A/B=par de atributos.
    # --------------------------------------------------------
    print("Rodando SC-WEAT estilo TCC/pares...")
    linhas_pares = []
    for atributo_A, atributo_B in PARES_CLASSICOS:
        A = conjuntos.get(atributo_A, [])
        B = conjuntos.get(atributo_B, [])
        if BALANCEAR_PARES_ATRIBUTOS:
            A, B = balancear_por_frequencia(A, B, vocab_freq)
        for alvo_nome in ["Feminino", "Masculino"]:
            W = conjuntos[alvo_nome]
            vals = valores_s(model, W, A, B)
            res = scweat_set(model, W, A, B)
            p, mu_perm, sd_perm = permutation_signflip_scweat(vals, N_PERMUTACOES, P_VALOR_BICAUDAL, SEED)
            linhas_pares.append({
                "atributo_A": atributo_A,
                "atributo_B": atributo_B,
                "alvo": alvo_nome,
                "n_W": len(W),
                "n_A": len(A),
                "n_B": len(B),
                "scweat_soma": res["scweat_soma"],
                "scweat_media": res["scweat_media"],
                "std_s": res["std_s"],
                "effect_size_d_assinado": res["effect_size_d"],
                "magnitude_do_efeito": abs(res["effect_size_d"]) if not pd.isna(res["effect_size_d"]) else np.nan,
                "p_value": p,
                "media_perm": mu_perm,
                "desvio_perm": sd_perm,
            })
    df_pares = pd.DataFrame(linhas_pares)
    df_pares.to_csv(f"{PREFIXO_SAIDA}_scweat_pares_genero.csv", index=False, encoding="utf-8-sig")
    grafico_scweat_pares_tcc(df_pares, f"{PREFIXO_SAIDA}_grafico_scweat_pares_tcc.png")
    tabela_scweat_pares_png(df_pares, f"{PREFIXO_SAIDA}_tabela_scweat_pares_tcc.png")

    # --------------------------------------------------------
    # 3) Tabela comparativa estilo segundo print.
    #    Aqui A=Masculino, B=Feminino; X/Y são pares de categorias.
    # --------------------------------------------------------
    print("Gerando tabela comparativa WEAT/SC-WEAT...")
    linhas_comp = []
    for X_nome, Y_nome in PARES_TABELA_COMPARATIVA:
        A_nome = "Masculino"
        B_nome = "Feminino"
        A = masculino_bal
        B = feminino_bal
        X = conjuntos.get(X_nome, [])
        Y = conjuntos.get(Y_nome, [])
        if BALANCEAR_PARES_ATRIBUTOS:
            X, Y = balancear_por_frequencia(X, Y, vocab_freq)
        if min(len(A), len(B), len(X), len(Y)) < 2:
            continue

        vals_x = valores_s(model, X, A, B)
        vals_y = valores_s(model, Y, A, B)
        res_x = scweat_set(model, X, A, B)
        res_y = scweat_set(model, Y, A, B)
        p_x, _, _ = permutation_signflip_scweat(vals_x, N_PERMUTACOES, P_VALOR_BICAUDAL, SEED)
        p_y, _, _ = permutation_signflip_scweat(vals_y, N_PERMUTACOES, P_VALOR_BICAUDAL, SEED)

        svals = precomputar_s(model, list(dict.fromkeys(X + Y)), A, B)
        weat_d = effect_size_weat_precomp(svals, X, Y)
        p_weat, _, _ = permutation_test_weat_precomp(svals, X, Y, N_PERMUTACOES, P_VALOR_BICAUDAL, SEED)

        linhas_comp.append({
            "A": A_nome,
            "B": B_nome,
            "X": X_nome,
            "Y": Y_nome,
            "corpus": caminho_dados.name,
            "SC-WEAT X": res_x["effect_size_d"],
            "SC-WEAT Y": res_y["effect_size_d"],
            "p SC-WEAT X": p_x,
            "p SC-WEAT Y": p_y,
            "WEAT effect size": weat_d,
            "p WEAT": p_weat,
            "SC-WEAT X fmt": fmt_num_sig(res_x["effect_size_d"], p_x, 2),
            "SC-WEAT Y fmt": fmt_num_sig(res_y["effect_size_d"], p_y, 2),
            "WEAT effect size fmt": fmt_num_sig(weat_d, p_weat, 2),
            "n_A": len(A),
            "n_B": len(B),
            "n_X": len(X),
            "n_Y": len(Y),
        })
    df_comp = pd.DataFrame(linhas_comp)
    df_comp.to_csv(f"{PREFIXO_SAIDA}_tabela_comparativa_weat_scweat.csv", index=False, encoding="utf-8-sig")
    tabela_comparativa_png(df_comp, f"{PREFIXO_SAIDA}_tabela_comparativa_weat_scweat.png")

    # --------------------------------------------------------
    # 4) Parâmetros e resumo.
    # --------------------------------------------------------
    parametros = pd.DataFrame([
        {"parametro": "arquivo_dados", "valor": str(caminho_dados), "justificativa": "arquivo usado para treinar Word2Vec"},
        {"parametro": "coluna_texto", "valor": coluna_texto, "justificativa": "coluna usada como letra/texto"},
        {"parametro": "VECTOR_SIZE", "valor": VECTOR_SIZE, "justificativa": "dimensão comum e estável para corpus pequeno/médio"},
        {"parametro": "WINDOW", "valor": WINDOW, "justificativa": "janela local padrão do Word2Vec"},
        {"parametro": "MIN_COUNT", "valor": MIN_COUNT, "justificativa": "mantém termos raros de alvos/atributos"},
        {"parametro": "SG", "valor": SG, "justificativa": "1=skip-gram, melhor para termos menos frequentes"},
        {"parametro": "EPOCHS", "valor": EPOCHS, "justificativa": "mais épocas para corpus menor"},
        {"parametro": "SEED", "valor": SEED, "justificativa": "reprodutibilidade"},
        {"parametro": "REMOVER_STOPWORDS", "valor": REMOVER_STOPWORDS, "justificativa": "remove ruído, preservando alvos/atributos"},
        {"parametro": "PRESERVAR_ACENTOS_NOS_TOKENS", "valor": PRESERVAR_ACENTOS_NOS_TOKENS, "justificativa": "evita juntar avó/avô"},
        {"parametro": "N_PERMUTACOES", "valor": N_PERMUTACOES, "justificativa": "estima p-valor por permutação"},
        {"parametro": "BALANCEAR_MASC_FEM", "valor": BALANCEAR_MASC_FEM, "justificativa": "igualar tamanho dos conjuntos masculino/feminino"},
        {"parametro": "BALANCEAR_PARES_ATRIBUTOS", "valor": BALANCEAR_PARES_ATRIBUTOS, "justificativa": "igualar tamanho dos pares de categorias"},
    ])
    parametros.to_csv(f"{PREFIXO_SAIDA}_parametros.csv", index=False, encoding="utf-8-sig")

    print("\nArquivos gerados:")
    for nome in [
        f"{PREFIXO_SAIDA}_scweat_chen.csv",
        f"{PREFIXO_SAIDA}_grafico_scweat_chen.png",
        f"{PREFIXO_SAIDA}_scweat_pares_genero.csv",
        f"{PREFIXO_SAIDA}_grafico_scweat_pares_tcc.png",
        f"{PREFIXO_SAIDA}_tabela_scweat_pares_tcc.png",
        f"{PREFIXO_SAIDA}_tabela_comparativa_weat_scweat.csv",
        f"{PREFIXO_SAIDA}_tabela_comparativa_weat_scweat.png",
        f"{PREFIXO_SAIDA}_auditoria_vocabulario.csv",
        f"{PREFIXO_SAIDA}_parametros.csv",
        f"{PREFIXO_SAIDA}_word2vec.model",
    ]:
        print("-", nome)


if __name__ == "__main__":
    main()
