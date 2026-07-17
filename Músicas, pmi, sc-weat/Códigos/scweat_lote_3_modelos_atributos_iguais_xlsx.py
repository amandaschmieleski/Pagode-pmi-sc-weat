# -*- coding: utf-8 -*-
r"""
SC-WEAT standalone em lote para 6 CSVs e 3 modelos de embedding:
Word2Vec, fastText e GloVe.

O script NÃO depende do código de PMI. As listas ALVOS e ATRIBUTOS foram
incorporadas aqui para que o SC-WEAT rode sozinho.

Arquivos esperados por padrão em D:\Downloads:
    merged_df_minusculo_sem_duplicadas.csv
    pagode_tudo_um_csv_com_letras.csv
    sertanejo_tudo_um_csv_com_letras.csv
    forro_tudo_um_csv_com_letras.csv
    funk_tudo_um_csv_com_letras.csv
    mpb_tudo_um_csv_com_letras.csv

Saída padrão:
    D:\Downloads\resultados_scweat_todos_modelos

Versão enxuta:
    Esta versão NÃO salva os CSVs individuais por execução:
    - *_scweat_chen.csv
    - *_scweat_pares_genero.csv
    - *_tabela_comparativa_weat_scweat.csv
    - *_auditoria_vocabulario.csv
    - *_parametros.csv

    Mantém os PNGs por execução, os embeddings salvos e os consolidados finais.

Dependências:
    pip install pandas numpy matplotlib gensim scipy

Observação sobre GloVe:
    gensim treina Word2Vec e fastText, mas não treina GloVe. Por isso este
    script inclui uma implementação simples de GloVe em NumPy. Ela é suficiente
    para rodar SC-WEAT sem alterar a fórmula do teste, mas pode ser mais lenta
    que bibliotecas otimizadas em C/C++.
"""

from __future__ import annotations

import csv
import itertools
import math
import random
import re
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from gensim.models import Word2Vec, FastText
except ImportError as exc:
    raise RuntimeError(
        "gensim não está instalado. Instale com:\n"
        "  pip install gensim scipy\n"
    ) from exc

# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

PASTA = Path(r"D:\Downloads")
PASTA_SAIDA_LOTE = PASTA / "resultados_scweat_todos_modelos"
PASTA_SAIDA_LOTE.mkdir(parents=True, exist_ok=True)

ARQUIVOS_ENTRADA = {
    "merged_df_minusculo": PASTA / "merged_df_minusculo_sem_duplicadas.csv",
    "pagode": PASTA / "pagode_tudo_um_csv_com_letras.csv",
    "sertanejo": PASTA / "sertanejo_tudo_um_csv_com_letras.csv",
    "forro": PASTA / "forro_tudo_um_csv_com_letras.csv",
    "funk": PASTA / "funk_tudo_um_csv_com_letras.csv",
    "mpb": PASTA / "mpb_tudo_um_csv_com_letras.csv",
}

MODELOS_EMBEDDING = ["word2vec", "glove", "fasttext"]

COLUNA_TEXTO_FIXA = None
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

USAR_PRIMEIRAS_N_LINHAS = None
PRESERVAR_ACENTOS_NOS_TOKENS = True
REMOVER_STOPWORDS = True

# Parâmetros compartilhados por Word2Vec e fastText.
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 1
SG = 1
EPOCHS_WORD2VEC = 50
EPOCHS_FASTTEXT = 50
WORKERS = 4
SEED = 42

# Parâmetros específicos do fastText.
FASTTEXT_MIN_N = 3
FASTTEXT_MAX_N = 6
FASTTEXT_BUCKET = 2_000_000

# Parâmetros específicos do GloVe local.
GLOVE_VECTOR_SIZE = VECTOR_SIZE
GLOVE_WINDOW = WINDOW
GLOVE_EPOCHS = 25
GLOVE_LEARNING_RATE = 0.05
GLOVE_X_MAX = 100.0
GLOVE_ALPHA = 0.75
GLOVE_MAX_VOCAB = 30_000
GLOVE_MAX_COOCS = None  # Ex.: 2_000_000 para limitar custo em máquina fraca.

# Permutações para p-valor.
N_PERMUTACOES = 1000
P_VALOR_BICAUDAL = True

# Balanceamento por frequência para comparações simétricas.
BALANCEAR_MASC_FEM = True
BALANCEAR_PARES_ATRIBUTOS = True

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
# LISTAS INCORPORADAS DO PMI: ALVOS E ATRIBUTOS
# ============================================================

ALVOS = {'Feminino': ['alcione',
              'alice',
              'aline',
              'avó',
              'avós',
              'avozinha',
              'babi',
              'baby',
              'baiana',
              'beth',
              'brasileira',
              'carolina',
              'claudia',
              'dama',
              'damas',
              'daminha',
              'dela',
              'delas',
              'dolores',
              'dona',
              'donas',
              'dora',
              'ela',
              'elas',
              'elka',
              'erika',
              'esposa',
              'esposas',
              'esposinha',
              'esposona',
              'estrangeira',
              'fabiana',
              'fêmea',
              'fêmeas',
              'filha',
              'filhas',
              'filhinha',
              'filhona',
              'gabriela',
              'garota',
              'garotas',
              'garotinha',
              'garotona',
              'inara',
              'iolanda',
              'irene',
              'irmã',
              'irmãs',
              'irmãzinha',
              'irmãzona',
              'jéssica',
              'julieta',
              'leandra',
              'leonor',
              'ludmilla',
              'luzia',
              'madame',
              'madames',
              'mãe',
              'mães',
              'mãezinha',
              'mãezona',
              'mainha',
              'mamãe',
              'mamães',
              'mamãezinha',
              'mamãezona',
              'mana',
              'manas',
              'maninha',
              'maria',
              'marilia',
              'menina',
              'meninas',
              'menininha',
              'meninona',
              'mina',
              'minas',
              'minazinha',
              'moça',
              'moças',
              'mocinha',
              'moçona',
              'mulata',
              'mulatinha',
              'mulher',
              'mulherada',
              'mulheres',
              'mulherona',
              'mulherzinha',
              'namorada',
              'namoradas',
              'namoradinha',
              'namoradona',
              'nega',
              'negona',
              'neguinha',
              'nora',
              'noras',
              'norinha',
              'novinha',
              'pagodeira',
              'pagodeiras',
              'pagodeirinha',
              'rainha',
              'rainhas',
              'rainhazinha',
              'rita',
              'rosa',
              'rosalina',
              'sabrina',
              'sara',
              'senhora',
              'senhoras',
              'senhorinha',
              'senhorita',
              'senhoritas',
              'sobrinha',
              'sobrinhas',
              'sobrinhinha',
              'solteira',
              'solteiras',
              'solteirinha',
              'solteirona',
              'tia',
              'tias',
              'tiazinha',
              'tiazona',
              'titia',
              'titias',
              'titiazinha',
              'vizinha',
              'vizinhas',
              'vizinhinha',
              'vizinhona',
              'vovó',
              'vovós',
              'vovozinha',
              'vovozona'],
 'Masculino': ['abel',
               'adão',
               'alexandre',
               'almir',
               'anderson',
               'anthony',
               'arlindo',
               'augusto',
               'avô',
               'avôs',
               'avozinho',
               'bebeto',
               'beto',
               'bill',
               'bob',
               'bruno',
               'caetano',
               'cantor',
               'cantorão',
               'cantores',
               'cantorzinho',
               'cara',
               'carão',
               'caras',
               'carinha',
               'carlos',
               'charlie',
               'chico',
               'cláudio',
               'companheirão',
               'companheirinho',
               'companheiro',
               'companheiros',
               'cosme',
               'dele',
               'deles',
               'dilsinho',
               'dono',
               'donos',
               'doutor',
               'doutorão',
               'doutores',
               'doutorzinho',
               'ele',
               'eles',
               'emanuel',
               'filhão',
               'filhinho',
               'filho',
               'filhos',
               'gabriel',
               'garotão',
               'garotinho',
               'garoto',
               'garotos',
               'genrinho',
               'genro',
               'genros',
               'guilherme',
               'gustavo',
               'homão',
               'homem',
               'homens',
               'homenzinho',
               'ícaro',
               'irmão',
               'irmãos',
               'irmãozão',
               'irmãozinho',
               'iuri',
               'james',
               'jhonny',
               'joão',
               'jorge',
               'josé',
               'junior',
               'leandro',
               'léo',
               'leonardo',
               'luiz',
               'machão',
               'machinho',
               'macho',
               'machos',
               'manão',
               'mané',
               'manés',
               'manezão',
               'manezinho',
               'maninho',
               'mano',
               'manoel',
               'manos',
               'marcelo',
               'marcinho',
               'maridão',
               'maridinho',
               'marido',
               'maridos',
               'mario',
               'marquinho',
               'marquinhos',
               'meninão',
               'menininho',
               'menino',
               'meninos',
               'mestrão',
               'mestre',
               'mestres',
               'mestrinho',
               'miguel',
               'mino',
               'minos',
               'mocinho',
               'moço',
               'moços',
               'molecão',
               'moleque',
               'moleques',
               'molequinho',
               'mulato',
               'namoradão',
               'namoradinho',
               'namorado',
               'namorados',
               'nathan',
               'negão',
               'nego',
               'neguinho',
               'netinho',
               'novo',
               'pablo',
               'pagodeirinho',
               'pagodeiro',
               'pagodeiros',
               'pai',
               'painho',
               'painhos',
               'pais',
               'paizão',
               'paizinho',
               'papai',
               'papais',
               'papaizão',
               'papaizinho',
               'patrão',
               'patrãozão',
               'patrãozinho',
               'patrões',
               'pedro',
               'poeta',
               'poetão',
               'poetas',
               'poetinha',
               'primão',
               'priminho',
               'primo',
               'primos',
               'rafael',
               'rapagão',
               'rapaz',
               'rapazes',
               'rapaziada',
               'rapazinho',
               'rei',
               'reis',
               'reizão',
               'reizinho',
               'ricardo',
               'rodriguinho',
               'romeu',
               'romeu',
               'romeus',
               'romeuzinho',
               'rubem',
               'safadão',
               'safadinho',
               'safado',
               'safados',
               'senhor',
               'senhorão',
               'senhores',
               'senhorzinho',
               'sergio',
               'sobrinhinho',
               'sobrinho',
               'sobrinhos',
               'sogrão',
               'sogrinho',
               'sogro',
               'sogros',
               'solteirão',
               'solteirinho',
               'solteiro',
               'solteiros',
               'suel',
               'thiago',
               'thiaguinho',
               'tião',
               'tio',
               'tios',
               'tiozão',
               'tiozinho',
               'titio',
               'vavá',
               'vinícius',
               'vitinho',
               'vizinhão',
               'vizinhinho',
               'vizinho',
               'vizinhos',
               'vovô',
               'vovôs',
               'vovozão',
               'vovozinho',
               'zé',
               'zeca']}

# ATRIBUTOS ajustados para ficar igual ao XLSX "Atributos e alvos - listas Chen e listas Amanda.xlsx".
ATRIBUTOS = {'Agradável': ['abraçar',
               'apegado',
               'certo',
               'especial',
               'incrível',
               'respeito',
               'sensacional',
               'sorrir',
               'união',
               'verdadeiro',
               'abraço',
               'adorar',
               'alegre',
               'alegria',
               'amado',
               'amar',
               'amigo',
               'amor',
               'anjo',
               'apaixonar',
               'bacana',
               'beijar',
               'beijo',
               'bem',
               'bom',
               'brilho',
               'brincalhão',
               'carinho',
               'carinhoso',
               'carisma',
               'céu',
               'cheirosa',
               'companheira',
               'confiar',
               'contente',
               'coração',
               'cuidar',
               'curar',
               'curtir',
               'decente',
               'desejo',
               'deusa',
               'diamante',
               'disposição',
               'doce',
               'elogiar',
               'empolgada',
               'escolhida',
               'espetacular',
               'estrela',
               'faceiro',
               'família',
               'famosa',
               'felicidade',
               'feliz',
               'férias',
               'fiel',
               'flor',
               'honesto',
               'honra',
               'inocente',
               'joia',
               'lealdade',
               'legal',
               'liberdade',
               'luz',
               'maneira',
               'maravilha',
               'maravilhoso',
               'medicinal',
               'milagre',
               'nascer',
               'nobre',
               'nota 10',
               'ouro',
               'paixão',
               'paraíso',
               'parceiro',
               'paz',
               'perfeição',
               'perfeito',
               'prazer',
               'presente',
               'puro',
               'radiante',
               'remédio',
               'responsa',
               'rico',
               'riqueza',
               'rir',
               'riso',
               'romance',
               'saudade',
               'saúde',
               'simpatia',
               'simpático',
               'sincero',
               'sonhar',
               'sonho',
               'sorriso',
               'sorte',
               'ternura'],
 'Desagradável': ['abandonar',
                  'abusar',
                  'adversidade',
                  'agonia',
                  'amargo',
                  'bagaceira',
                  'bagunça',
                  'bandido',
                  'barraqueira',
                  'barreira',
                  'bipolar',
                  'bomba',
                  'brigar',
                  'castigo',
                  'chato',
                  'chumbinho',
                  'ciumenta',
                  'congelar',
                  'crime',
                  'criminoso',
                  'delator',
                  'desconfiar',
                  'difícil',
                  'doença',
                  'doer',
                  'doido',
                  'dor',
                  'droga',
                  'enganar',
                  'exagerada',
                  'fácil',
                  'fanático',
                  'fatal',
                  'ferir',
                  'frio',
                  'fuleiro',
                  'fúria',
                  'gelado',
                  'granada',
                  'guerra',
                  'implicar',
                  'indigesta',
                  'infiel',
                  'ingrata',
                  'inimigo',
                  'invejar',
                  'irresponsável',
                  'ladrão',
                  'ligeira',
                  'lobo',
                  'luto',
                  'machucar',
                  'magoar',
                  'mal',
                  'malandro',
                  'maldosa',
                  'maloqueiro',
                  'maltratar',
                  'maluco',
                  'malvada',
                  'mandona',
                  'manhosa',
                  'maroto',
                  'matar',
                  'mau',
                  'mentir',
                  'mentira',
                  'morrer',
                  'morte',
                  'ódio',
                  'ofender',
                  'patricinha',
                  'pavor',
                  'perversa',
                  'piranha',
                  'piriguete',
                  'pobre',
                  'pobreza',
                  'prisão',
                  'problema',
                  'puta',
                  'rancor',
                  'ruim',
                  'safada',
                  'sapeca',
                  'sofrer',
                  'sofrimento',
                  'solidão',
                  'soltinha',
                  'sombrio',
                  'sozinho',
                  'sujeira',
                  'terrível',
                  'tormento',
                  'trair',
                  'triste',
                  'tristeza',
                  'vacilona',
                  'vagabundo',
                  'veneno',
                  'vulgar',
                  'xingar',
                  'zangado'],
 'Aparência': ['alto',
               'baixo',
               'boca',
               'bombado',
               'bonito',
               'branco',
               'cabelo',
               'calor',
               'cheia',
               'cheiro',
               'cigana',
               'cigano',
               'cinderela',
               'comprida',
               'corpo',
               'donzela',
               'elegante',
               'estilosa',
               'excitar',
               'feio',
               'formosa',
               'gata',
               'gatinha',
               'gato',
               'gostoso',
               'lindo',
               'loiro',
               'magra',
               'morena',
               'musa',
               'negro',
               'nua',
               'olhar',
               'olhos',
               'peituda',
               'pele',
               'perfume',
               'pretinho',
               'preto',
               'princesa',
               'provocar',
               'quente',
               'rebolar',
               'roupa',
               'saliente',
               'sedução',
               'sedutor',
               'seduzir',
               'sensual',
               'sereia',
               'sexy',
               'simples',
               'tanajura',
               'tentação',
               'tesão',
               'turbinada'],
 'Inteligência': ['adaptar',
                  'aprender',
                  'atualizado',
                  'brilhante',
                  'compreender',
                  'conhecer',
                  'educado',
                  'engraçado',
                  'ensinar',
                  'entender',
                  'esperto',
                  'estudar',
                  'estudioso',
                  'imaginar',
                  'inteligente',
                  'inventar',
                  'lógico',
                  'pensar',
                  'perceber',
                  'refletir',
                  'sábio',
                  'sagaz'],
 'Força': ['atitude',
           'campeão',
           'comandar',
           'competidor',
           'confiança',
           'controle',
           'coragem',
           'dominar',
           'força',
           'forte',
           'guerreiro',
           'independente',
           'liderança',
           'lutar',
           'maduro',
           'ousadia',
           'ousado',
           'potência',
           'proteger',
           'protetor',
           'reprodutor',
           'seguro',
           'valente',
           'vencedor',
           'vencer',
           'vitória'],
 'Fraqueza': ['ansioso',
              'carente',
              'ceder',
              'chorar',
              'covarde',
              'defeito',
              'dependente',
              'deprimido',
              'derrota',
              'dificuldade',
              'doente',
              'errar',
              'fraco',
              'fraqueza',
              'inseguro',
              'medo',
              'nervoso',
              'otário',
              'perder',
              'quieto',
              'timidez',
              'tolo']}

ORDEM_CATEGORIAS = ['Agradável', 'Desagradável', 'Aparência', 'Inteligência', 'Força', 'Fraqueza']

# ============================================================
# LEITURA ROBUSTA DO CSV
# ============================================================


def normalizar_nome_coluna(nome: str) -> str:
    s = unicodedata.normalize("NFKD", str(nome).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def detectar_encoding_e_separador(caminho: Path) -> tuple[str, str]:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    amostra_bytes = caminho.read_bytes()[:200_000]
    ultimo_erro = None

    for enc in encodings:
        try:
            texto = amostra_bytes.decode(enc)
            try:
                dialect = csv.Sniffer().sniff(texto, delimiters=",;\t|")
                return enc, dialect.delimiter
            except Exception:
                candidatos = [",", ";", "\t", "|"]
                sep = max(candidatos, key=lambda s: texto.count(s))
                return enc, sep
        except Exception as exc:
            ultimo_erro = exc

    raise RuntimeError(f"Não consegui detectar encoding de {caminho}: {ultimo_erro}")


def ler_dataframe(caminho: Path) -> pd.DataFrame:
    enc, sep = detectar_encoding_e_separador(caminho)
    print(f"Lendo: {caminho.name} | encoding={enc} | sep={repr(sep)}")
    try:
        df = pd.read_csv(caminho, encoding=enc, sep=sep, dtype=str, low_memory=False)
    except Exception:
        df = pd.read_csv(
            caminho,
            encoding=enc,
            sep=sep,
            dtype=str,
            engine="python",
            on_bad_lines="warn",
        )

    if USAR_PRIMEIRAS_N_LINHAS is not None:
        df = df.head(USAR_PRIMEIRAS_N_LINHAS).copy()

    return df


def detectar_coluna_texto(df: pd.DataFrame) -> str:
    if COLUNA_TEXTO_FIXA and COLUNA_TEXTO_FIXA in df.columns:
        return COLUNA_TEXTO_FIXA

    mapa = {normalizar_nome_coluna(c): c for c in df.columns}
    for c in POSSIVEIS_COLUNAS_TEXTO:
        chave = normalizar_nome_coluna(c)
        if chave in mapa:
            return mapa[chave]

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


def coletar_palavras_interesse(alvos: dict, atributos: dict) -> set[str]:
    palavras = set()
    for lista in alvos.values():
        palavras.update(normalizar_lista_palavras(lista))
    for lista in atributos.values():
        palavras.update(normalizar_lista_palavras(lista))
    return palavras


def carregar_stopwords(palavras_interesse: set[str]) -> set[str]:
    if not REMOVER_STOPWORDS:
        return set()
    try:
        from spacy.lang.pt.stop_words import STOP_WORDS
        stop = {normalizar_token(w) for w in STOP_WORDS if normalizar_token(w)}
    except Exception:
        stop = {normalizar_token(w) for w in STOPWORDS_PT_MIN if normalizar_token(w)}
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


def tokenizar_dataframe(df: pd.DataFrame, coluna_texto: str, stopwords: set[str]) -> list[list[str]]:
    sentencas = []
    for texto in df[coluna_texto].fillna("").astype(str):
        toks = tokenizar_texto(texto, stopwords)
        if toks:
            sentencas.append(toks)
    return sentencas

# ============================================================
# ABSTRAÇÕES DE MODELO DE EMBEDDING
# ============================================================


class GensimEmbeddingSpace:
    def __init__(self, nome_modelo: str, model):
        self.nome_modelo = nome_modelo
        self.model = model
        self.wv = model.wv
        self.key_to_index = self.wv.key_to_index

    def has(self, word: str) -> bool:
        return word in self.key_to_index

    def similarity(self, w1: str, w2: str) -> float:
        return float(self.wv.similarity(w1, w2))

    def freq(self, word: str) -> int:
        if word not in self.key_to_index:
            return 0
        try:
            return int(self.wv.get_vecattr(word, "count"))
        except Exception:
            return 1

    def vocab_freq(self) -> dict[str, int]:
        return {w: self.freq(w) for w in self.key_to_index}

    def save(self, caminho: Path):
        self.model.save(str(caminho))


class NumpyEmbeddingSpace:
    def __init__(self, nome_modelo: str, words: list[str], vectors: np.ndarray, counts: dict[str, int]):
        self.nome_modelo = nome_modelo
        self.words = words
        self.key_to_index = {w: i for i, w in enumerate(words)}
        self.vectors = vectors.astype(np.float32)
        self.counts = counts
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.norm_vectors = self.vectors / norms

    def has(self, word: str) -> bool:
        return word in self.key_to_index

    def similarity(self, w1: str, w2: str) -> float:
        i = self.key_to_index[w1]
        j = self.key_to_index[w2]
        return float(np.dot(self.norm_vectors[i], self.norm_vectors[j]))

    def freq(self, word: str) -> int:
        return int(self.counts.get(word, 0))

    def vocab_freq(self) -> dict[str, int]:
        return {w: self.freq(w) for w in self.key_to_index}

    def save(self, caminho: Path):
        np.savez_compressed(
            caminho,
            words=np.array(self.words, dtype=object),
            vectors=self.vectors,
            counts=np.array([self.counts.get(w, 0) for w in self.words], dtype=np.int64),
        )

# ============================================================
# TREINO DOS MODELOS
# ============================================================


def treinar_word2vec(sentencas: list[list[str]]) -> GensimEmbeddingSpace:
    model = Word2Vec(
        sentences=sentencas,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        workers=WORKERS,
        seed=SEED,
        epochs=EPOCHS_WORD2VEC,
    )
    return GensimEmbeddingSpace("word2vec", model)


def treinar_fasttext(sentencas: list[list[str]]) -> GensimEmbeddingSpace:
    model = FastText(
        sentences=sentencas,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        workers=WORKERS,
        seed=SEED,
        epochs=EPOCHS_FASTTEXT,
        min_n=FASTTEXT_MIN_N,
        max_n=FASTTEXT_MAX_N,
        bucket=FASTTEXT_BUCKET,
    )
    return GensimEmbeddingSpace("fasttext", model)


def montar_vocabulario_glove(sentencas: list[list[str]], palavras_interesse: set[str]) -> tuple[list[str], dict[str, int]]:
    cont = Counter(tok for sent in sentencas for tok in sent)
    presentes_interesse = [w for w in palavras_interesse if cont.get(w, 0) > 0]

    palavras_ordenadas = [w for w, _ in cont.most_common()]
    vocab = []
    vistos = set()

    # Garante que alvos/atributos presentes no corpus entrem no vocabulário.
    for w in sorted(presentes_interesse, key=lambda x: (-cont[x], x)):
        if w not in vistos:
            vistos.add(w)
            vocab.append(w)

    for w in palavras_ordenadas:
        if len(vocab) >= GLOVE_MAX_VOCAB:
            break
        if w not in vistos:
            vistos.add(w)
            vocab.append(w)

    return vocab, dict(cont)


def construir_coocorrencias_glove(sentencas: list[list[str]], word_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cooc = defaultdict(float)

    for sent in sentencas:
        ids = [word_to_idx[w] for w in sent if w in word_to_idx]
        for i, wi in enumerate(ids):
            ini = max(0, i - GLOVE_WINDOW)
            fim = min(len(ids), i + GLOVE_WINDOW + 1)
            for j in range(ini, fim):
                if i == j:
                    continue
                wj = ids[j]
                distancia = abs(i - j)
                if distancia == 0:
                    continue
                cooc[(wi, wj)] += 1.0 / distancia

    items = list(cooc.items())
    if GLOVE_MAX_COOCS is not None and len(items) > GLOVE_MAX_COOCS:
        print(f"Limitando coocorrências GloVe: {len(items)} -> {GLOVE_MAX_COOCS}")
        items.sort(key=lambda kv: kv[1], reverse=True)
        items = items[:GLOVE_MAX_COOCS]

    if not items:
        raise ValueError("Nenhuma coocorrência foi gerada para o GloVe.")

    i_idx = np.array([k[0] for k, _ in items], dtype=np.int32)
    j_idx = np.array([k[1] for k, _ in items], dtype=np.int32)
    xij = np.array([v for _, v in items], dtype=np.float32)
    return i_idx, j_idx, xij


def treinar_glove(sentencas: list[list[str]], palavras_interesse: set[str]) -> NumpyEmbeddingSpace:
    vocab, counts = montar_vocabulario_glove(sentencas, palavras_interesse)
    if len(vocab) < 2:
        raise ValueError("Vocabulário insuficiente para treinar GloVe.")

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    print(f"GloVe | vocabulário={len(vocab)} | construindo coocorrências...")
    i_idx, j_idx, xij = construir_coocorrencias_glove(sentencas, word_to_idx)
    print(f"GloVe | pares de coocorrência={len(xij)} | treinando...")

    rng = np.random.default_rng(SEED)
    n_vocab = len(vocab)
    dim = GLOVE_VECTOR_SIZE

    W = rng.normal(0, 0.01, size=(n_vocab, dim)).astype(np.float32)
    C = rng.normal(0, 0.01, size=(n_vocab, dim)).astype(np.float32)
    bW = np.zeros(n_vocab, dtype=np.float32)
    bC = np.zeros(n_vocab, dtype=np.float32)

    grad_sq_W = np.ones_like(W, dtype=np.float32)
    grad_sq_C = np.ones_like(C, dtype=np.float32)
    grad_sq_bW = np.ones_like(bW, dtype=np.float32)
    grad_sq_bC = np.ones_like(bC, dtype=np.float32)

    weights = np.where(xij < GLOVE_X_MAX, (xij / GLOVE_X_MAX) ** GLOVE_ALPHA, 1.0).astype(np.float32)
    log_x = np.log(xij).astype(np.float32)
    ordem = np.arange(len(xij))

    for epoca in range(1, GLOVE_EPOCHS + 1):
        rng.shuffle(ordem)
        custo = 0.0
        for k in ordem:
            i = i_idx[k]
            j = j_idx[k]
            peso = weights[k]
            diff = float(np.dot(W[i], C[j]) + bW[i] + bC[j] - log_x[k])
            fdiff = peso * diff
            custo += 0.5 * peso * diff * diff

            grad_w = fdiff * C[j]
            grad_c = fdiff * W[i]
            grad_bw = fdiff
            grad_bc = fdiff

            W[i] -= (GLOVE_LEARNING_RATE * grad_w) / np.sqrt(grad_sq_W[i])
            C[j] -= (GLOVE_LEARNING_RATE * grad_c) / np.sqrt(grad_sq_C[j])
            bW[i] -= (GLOVE_LEARNING_RATE * grad_bw) / math.sqrt(float(grad_sq_bW[i]))
            bC[j] -= (GLOVE_LEARNING_RATE * grad_bc) / math.sqrt(float(grad_sq_bC[j]))

            grad_sq_W[i] += grad_w * grad_w
            grad_sq_C[j] += grad_c * grad_c
            grad_sq_bW[i] += grad_bw * grad_bw
            grad_sq_bC[j] += grad_bc * grad_bc

        print(f"GloVe | época {epoca:02d}/{GLOVE_EPOCHS} | custo médio={custo / len(xij):.6f}")

    vectors = W + C
    return NumpyEmbeddingSpace("glove", vocab, vectors, counts)


def treinar_modelo(nome_modelo: str, sentencas: list[list[str]], palavras_interesse: set[str]):
    if nome_modelo == "word2vec":
        return treinar_word2vec(sentencas)
    if nome_modelo == "fasttext":
        return treinar_fasttext(sentencas)
    if nome_modelo == "glove":
        return treinar_glove(sentencas, palavras_interesse)
    raise ValueError(f"Modelo desconhecido: {nome_modelo}")

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


def preparar_conjuntos_para_modelo(embedding, alvos: dict, atributos: dict) -> tuple[dict[str, list[str]], pd.DataFrame]:
    vocab_freq = embedding.vocab_freq()
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

    return conjuntos, pd.DataFrame(auditoria).drop_duplicates()


def balancear_por_frequencia(a: list[str], b: list[str], vocab_freq: dict[str, int]) -> tuple[list[str], list[str]]:
    n = min(len(a), len(b))
    a_ord = sorted(a, key=lambda w: (-vocab_freq.get(w, 0), w))[:n]
    b_ord = sorted(b, key=lambda w: (-vocab_freq.get(w, 0), w))[:n]
    return sorted(a_ord), sorted(b_ord)

# ============================================================
# CÁLCULOS WEAT / SC-WEAT
# ============================================================


def s_palavra(embedding, w: str, A: list[str], B: list[str]) -> float:
    if not A or not B:
        return float("nan")
    media_a = float(np.mean([embedding.similarity(w, a) for a in A]))
    media_b = float(np.mean([embedding.similarity(w, b) for b in B]))
    return media_a - media_b


def valores_s(embedding, W: list[str], A: list[str], B: list[str]) -> np.ndarray:
    vals = np.array([s_palavra(embedding, w, A, B) for w in W], dtype=float)
    return vals[~np.isnan(vals)]


def scweat_set(embedding, W: list[str], A: list[str], B: list[str]) -> dict:
    vals = valores_s(embedding, W, A, B)
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


def precomputar_s(embedding, palavras: list[str], A: list[str], B: list[str]) -> dict[str, float]:
    return {w: s_palavra(embedding, w, A, B) for w in palavras}


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


def grafico_scweat_chen(df: pd.DataFrame, nome_saida: Path):
    import matplotlib.pyplot as plt

    plot = df.set_index("categoria").reindex(ORDEM_CATEGORIAS).reset_index()
    labels = [LABELS_CHEN.get(c, c) for c in plot["categoria"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(plot))
    ax.bar(x, plot["effect_size_d"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", linestyle="-", alpha=0.35)
    ax.set_xlabel("Target Set", fontsize=12)
    ax.set_ylabel("Effect Size", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
    ax.set_title("SC-WEAT effect size", fontsize=13)

    if not plot["effect_size_d"].isna().all():
        minv = float(np.nanmin(plot["effect_size_d"]))
        maxv = float(np.nanmax(plot["effect_size_d"]))
    else:
        minv, maxv = -0.1, 0.1
    margem = max(0.12, (maxv - minv) * 0.18)
    ax.set_ylim(min(-0.1, minv - margem), max(0.1, maxv + margem))

    plt.tight_layout()
    plt.savefig(nome_saida, dpi=300, bbox_inches="tight")
    plt.close()


def grafico_scweat_pares_tcc(df: pd.DataFrame, nome_saida: Path):
    import matplotlib.pyplot as plt

    ordem_labels = [f"{a}/{b}" for a, b in PARES_CLASSICOS]
    plot = df.copy()
    plot["par_label"] = plot["atributo_A"] + "/" + plot["atributo_B"]
    pivot = plot.pivot(index="par_label", columns="alvo", values="magnitude_do_efeito")
    pivot = pivot.reindex(ordem_labels)

    x = np.arange(len(pivot.index))
    largura = 0.36
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.bar(x - largura / 2, pivot.get("Masculino", pd.Series(index=pivot.index, dtype=float)), largura, label="Masculino")
    ax.bar(x + largura / 2, pivot.get("Feminino", pd.Series(index=pivot.index, dtype=float)), largura, label="Feminino")
    ax.set_title("Comparativo do SC-WEAT para Gêneros Masculino e Feminino", fontsize=13)
    ax.set_xlabel("Categoria")
    ax.set_ylabel("Magnitude do Efeito")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=50, ha="right")
    ax.legend(title="Gênero")
    maxv = float(np.nanmax(pivot.values)) if not np.isnan(pivot.values).all() else 0.1
    ax.set_ylim(0, max(0.1, maxv * 1.15))
    plt.tight_layout()
    plt.savefig(nome_saida, dpi=300, bbox_inches="tight")
    plt.close()


def tabela_scweat_pares_png(df: pd.DataFrame, nome_saida: Path):
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


def tabela_comparativa_png(df: pd.DataFrame, nome_saida: Path):
    import matplotlib.pyplot as plt

    colunas = ["A", "B", "X", "Y", "corpus", "modelo", "SC-WEAT X", "SC-WEAT Y", "WEAT\neffect size"]
    linhas = []
    for _, row in df.iterrows():
        linhas.append([
            row["A"], row["B"], row["X"], row["Y"], row["corpus"], row["modelo"],
            row["SC-WEAT X fmt"], row["SC-WEAT Y fmt"], row["WEAT effect size fmt"],
        ])

    altura = max(3.4, 0.42 * (len(linhas) + 2))
    fig, ax = plt.subplots(figsize=(14.5, altura))
    ax.axis("off")
    table = ax.table(cellText=linhas, colLabels=colunas, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.42)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Tabela comparativa WEAT/SC-WEAT", fontsize=13, pad=18)
    fig.text(0.02, 0.02, "*p < 0,10; **p < 0,05", fontsize=9)
    plt.tight_layout()
    plt.savefig(nome_saida, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# EXECUÇÃO DO SC-WEAT PARA UM CSV + UM MODELO
# ============================================================


def montar_prefixo_saida(nome_csv: str, nome_modelo: str) -> Path:
    return PASTA_SAIDA_LOTE / f"{nome_csv}_{nome_modelo}"


def caminho_modelo(prefixo_saida: Path, nome_modelo: str) -> Path:
    if nome_modelo in {"word2vec", "fasttext"}:
        return Path(f"{prefixo_saida}_{nome_modelo}.model")
    if nome_modelo == "glove":
        return Path(f"{prefixo_saida}_glove.npz")
    return Path(f"{prefixo_saida}_{nome_modelo}.model")


def rodar_scweat_para_embedding(
    embedding,
    nome_csv: str,
    caminho_csv: Path,
    coluna_texto: str,
    qtd_documentos: int,
    prefixo_saida: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    vocab_freq = embedding.vocab_freq()
    conjuntos, df_auditoria = preparar_conjuntos_para_modelo(embedding, ALVOS, ATRIBUTOS)
    # Versão enxuta: calcula a auditoria apenas internamente, mas não salva
    # *_auditoria_vocabulario.csv por execução.
    # Os resultados consolidados finais preservam as medidas necessárias para análise.

    masculino = conjuntos["Masculino"]
    feminino = conjuntos["Feminino"]
    if BALANCEAR_MASC_FEM:
        masculino_bal, feminino_bal = balancear_por_frequencia(masculino, feminino, vocab_freq)
    else:
        masculino_bal, feminino_bal = masculino, feminino

    # 1) SC-WEAT Chen: W=categoria, A=Masculino, B=Feminino.
    linhas_chen = []
    for categoria in ORDEM_CATEGORIAS:
        W = conjuntos.get(categoria, [])
        vals = valores_s(embedding, W, masculino_bal, feminino_bal)
        res = scweat_set(embedding, W, masculino_bal, feminino_bal)
        p, mu_perm, sd_perm = permutation_signflip_scweat(vals, N_PERMUTACOES, P_VALOR_BICAUDAL, SEED)
        linhas_chen.append({
            "csv_origem": nome_csv,
            "modelo": embedding.nome_modelo,
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
    grafico_scweat_chen(df_chen, Path(f"{prefixo_saida}_grafico_scweat_chen.png"))

    # 2) SC-WEAT estilo TCC: W=Feminino/Masculino; A/B=par de atributos.
    linhas_pares = []
    for atributo_A, atributo_B in PARES_CLASSICOS:
        A = conjuntos.get(atributo_A, [])
        B = conjuntos.get(atributo_B, [])
        if BALANCEAR_PARES_ATRIBUTOS:
            A, B = balancear_por_frequencia(A, B, vocab_freq)
        for alvo_nome in ["Feminino", "Masculino"]:
            W = conjuntos[alvo_nome]
            vals = valores_s(embedding, W, A, B)
            res = scweat_set(embedding, W, A, B)
            p, mu_perm, sd_perm = permutation_signflip_scweat(vals, N_PERMUTACOES, P_VALOR_BICAUDAL, SEED)
            linhas_pares.append({
                "csv_origem": nome_csv,
                "modelo": embedding.nome_modelo,
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
    grafico_scweat_pares_tcc(df_pares, Path(f"{prefixo_saida}_grafico_scweat_pares_tcc.png"))
    tabela_scweat_pares_png(df_pares, Path(f"{prefixo_saida}_tabela_scweat_pares_tcc.png"))

    # 3) Tabela comparativa WEAT/SC-WEAT.
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

        vals_x = valores_s(embedding, X, A, B)
        vals_y = valores_s(embedding, Y, A, B)
        res_x = scweat_set(embedding, X, A, B)
        res_y = scweat_set(embedding, Y, A, B)
        p_x, _, _ = permutation_signflip_scweat(vals_x, N_PERMUTACOES, P_VALOR_BICAUDAL, SEED)
        p_y, _, _ = permutation_signflip_scweat(vals_y, N_PERMUTACOES, P_VALOR_BICAUDAL, SEED)

        svals = precomputar_s(embedding, list(dict.fromkeys(X + Y)), A, B)
        weat_d = effect_size_weat_precomp(svals, X, Y)
        p_weat, _, _ = permutation_test_weat_precomp(svals, X, Y, N_PERMUTACOES, P_VALOR_BICAUDAL, SEED)

        linhas_comp.append({
            "csv_origem": nome_csv,
            "modelo": embedding.nome_modelo,
            "A": A_nome,
            "B": B_nome,
            "X": X_nome,
            "Y": Y_nome,
            "corpus": caminho_csv.name,
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
    tabela_comparativa_png(df_comp, Path(f"{prefixo_saida}_tabela_comparativa_weat_scweat.png"))

    # 4) Versão enxuta: não salva *_parametros.csv por execução.
    # Os parâmetros gerais do lote são salvos uma única vez em parametros_gerais_lote.csv.

    return df_chen, df_pares, df_comp

# ============================================================
# EXECUÇÃO EM LOTE
# ============================================================


def main_lote():
    random.seed(SEED)
    np.random.seed(SEED)

    palavras_interesse = coletar_palavras_interesse(ALVOS, ATRIBUTOS)
    stopwords = carregar_stopwords(palavras_interesse)

    resultados_chen = []
    resultados_pares = []
    resultados_comp = []
    relatorio_execucoes = []

    print("=" * 80)
    print("RODANDO SC-WEAT ENXUTO PARA 6 CSVs E 3 MODELOS: Word2Vec, GloVe e fastText")
    print("=" * 80)
    print(f"Pasta de saída: {PASTA_SAIDA_LOTE}")

    for nome_csv, caminho_csv in ARQUIVOS_ENTRADA.items():
        if not caminho_csv.exists():
            print(f"\n[AVISO] Arquivo não encontrado: {caminho_csv}")
            for nome_modelo in MODELOS_EMBEDDING:
                relatorio_execucoes.append({
                    "csv_origem": nome_csv,
                    "modelo": nome_modelo,
                    "caminho_csv": str(caminho_csv),
                    "status": "arquivo_nao_encontrado",
                    "erro": "",
                    "segundos": np.nan,
                })
            continue

        try:
            print("\n" + "#" * 80)
            print(f"CSV: {nome_csv}")
            print(f"Arquivo: {caminho_csv}")
            print("#" * 80)

            df = ler_dataframe(caminho_csv)
            coluna_texto = detectar_coluna_texto(df)
            print(f"Coluna de texto usada: {coluna_texto}")

            sentencas = tokenizar_dataframe(df, coluna_texto, stopwords)
            if not sentencas:
                raise ValueError("Nenhuma sentença/token foi gerado. Confira a coluna de letras.")
            print(f"Documentos com tokens: {len(sentencas)} de {len(df)}")

        except Exception as erro_csv:
            print(f"\n[ERRO] Falhou ao preparar CSV: {nome_csv}")
            print(str(erro_csv))
            traceback.print_exc()
            for nome_modelo in MODELOS_EMBEDDING:
                relatorio_execucoes.append({
                    "csv_origem": nome_csv,
                    "modelo": nome_modelo,
                    "caminho_csv": str(caminho_csv),
                    "status": "erro_preparacao_csv",
                    "erro": str(erro_csv),
                    "segundos": np.nan,
                })
            continue

        for nome_modelo in MODELOS_EMBEDDING:
            inicio = time.time()
            prefixo_saida = montar_prefixo_saida(nome_csv, nome_modelo)
            print("\n" + "=" * 80)
            print(f"CSV: {nome_csv} | Modelo: {nome_modelo}")
            print("=" * 80)

            try:
                print(f"Treinando {nome_modelo}...")
                embedding = treinar_modelo(nome_modelo, sentencas, palavras_interesse)

                caminho_emb = caminho_modelo(prefixo_saida, nome_modelo)
                embedding.save(caminho_emb)
                print(f"Embedding salvo em: {caminho_emb}")

                df_chen, df_pares, df_comp = rodar_scweat_para_embedding(
                    embedding=embedding,
                    nome_csv=nome_csv,
                    caminho_csv=caminho_csv,
                    coluna_texto=coluna_texto,
                    qtd_documentos=len(df),
                    prefixo_saida=prefixo_saida,
                )

                resultados_chen.append(df_chen)
                resultados_pares.append(df_pares)
                resultados_comp.append(df_comp)

                segundos = time.time() - inicio
                relatorio_execucoes.append({
                    "csv_origem": nome_csv,
                    "modelo": nome_modelo,
                    "caminho_csv": str(caminho_csv),
                    "status": "ok",
                    "erro": "",
                    "segundos": segundos,
                })
                print(f"[OK] Finalizado: {nome_csv} | {nome_modelo} | {segundos:.1f}s")

            except Exception as erro:
                segundos = time.time() - inicio
                print(f"\n[ERRO] Falhou: {nome_csv} | {nome_modelo}")
                print(str(erro))
                traceback.print_exc()
                relatorio_execucoes.append({
                    "csv_origem": nome_csv,
                    "modelo": nome_modelo,
                    "caminho_csv": str(caminho_csv),
                    "status": "erro",
                    "erro": str(erro),
                    "segundos": segundos,
                })

    # Consolidados.
    caminho_relatorio = PASTA_SAIDA_LOTE / "relatorio_execucoes.csv"
    pd.DataFrame(relatorio_execucoes).to_csv(caminho_relatorio, index=False, encoding="utf-8-sig")

    if resultados_chen:
        pd.concat(resultados_chen, ignore_index=True).to_csv(
            PASTA_SAIDA_LOTE / "resumo_consolidado_scweat_chen_todos_csvs_modelos.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if resultados_pares:
        pd.concat(resultados_pares, ignore_index=True).to_csv(
            PASTA_SAIDA_LOTE / "resumo_consolidado_scweat_pares_todos_csvs_modelos.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if resultados_comp:
        pd.concat(resultados_comp, ignore_index=True).to_csv(
            PASTA_SAIDA_LOTE / "resumo_consolidado_tabela_comparativa_todos_csvs_modelos.csv",
            index=False,
            encoding="utf-8-sig",
        )

    parametros_gerais = pd.DataFrame([
        {"item": "csvs", "valor": len(ARQUIVOS_ENTRADA)},
        {"item": "modelos", "valor": ", ".join(MODELOS_EMBEDDING)},
        {"item": "execucoes_previstas", "valor": len(ARQUIVOS_ENTRADA) * len(MODELOS_EMBEDDING)},
        {"item": "outputs_por_execucao", "valor": 5},
        {"item": "outputs_por_execucao_descricao", "valor": "4 PNGs/figuras + 1 modelo salvo = 5; CSVs individuais, auditoria e parametros por execucao nao sao salvos"},
        {"item": "outputs_consolidados", "valor": 5},
        {"item": "outputs_totais_se_tudo_rodar", "valor": len(ARQUIVOS_ENTRADA) * len(MODELOS_EMBEDDING) * 5 + 5},
    ])
    parametros_gerais.to_csv(PASTA_SAIDA_LOTE / "parametros_gerais_lote.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 80)
    print(f"Relatório: {caminho_relatorio}")
    print(f"Pasta de saída: {PASTA_SAIDA_LOTE}")


if __name__ == "__main__":
    main_lote()
