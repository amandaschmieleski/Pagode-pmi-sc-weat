import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAÇÕES
# ============================================================

CAMINHO_CSV = "merged_df.txt"
COLUNA_TEXTO = "Letra da Música"
COLUNA_MUSICA = "Nome da Música"
COLUNA_ARTISTA = "Artista"

USAR_PRIMEIRAS_N_MUSICAS = None
JANELA_CONTEXTO = 4
REMOVER_ALVO_DO_CONTEXTO = True

# NÃO deduplicar contextos: cada ocorrência de alvo gera um contexto.
DEDUPLICAR_CONTEXTOS = False

EPSILON = 0.5
MIN_OCORRENCIA_ATRIBUTO = 1

MODELO_SPACY = "pt_core_news_sm"
PREFIXO_SAIDA = "COMPLETO_146k_pmi_14_junho"

# ============================================================
# LISTAS
# ============================================================

ALVOS = {'Feminino': [
                         'alcione',
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
                         'vovozona',
                     ],
 'Masculino': [
                  'abel',
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
                  'zeca',
              ]}

ATRIBUTOS = {
    'Agradável': [
                      'abraçar',
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
                      'apegado',
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
                      'certo',
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
                      'especial',
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
                      'incrível',
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
                      'respeito',
                      'responsa',
                      'rico',
                      'riqueza',
                      'rir',
                      'riso',
                      'romance',
                      'saudade',
                      'saúde',
                      'sensacional',
                      'simpatia',
                      'simpático',
                      'sincero',
                      'sonhar',
                      'sonho',
                      'sorrir',
                      'sorriso',
                      'sorte',
                      'ternura',
                      'união',
                      'verdadeiro',
                  ],
    'Aparência': [
                      'alto',
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
                      'turbinada',
                  ],
    'Desagradável': [
                         'abandonar',
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
                         'zangado',
                     ],
    'Força': [
                  'atitude',
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
                  'vitória',
              ],
    'Fraqueza': [
                    'ansioso',
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
                    'tolo',
                ],
    'Inteligência': [
                         'adaptar',
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
                         'sagaz',
                     ]
}

ORDEM_CATEGORIAS = ['Agradável', 'Desagradável', 'Aparência', 'Inteligência', 'Força', 'Fraqueza']

# Palavras em que o lema automático do spaCy costuma criar ruído semântico.
# Ex.: nega -> negar; mulherão -> mulher; linda -> lindar; lindo -> lir.
ATRIBUTOS_SEM_LEMA_AUTOMATICO = {
  #  "nega", "negão", "nego", "negona", "negra", "negro", "neguinha", "neguinho",
   # "preta", "pretinha", "pretinho", "preto",
  #  "mulata", "mulatinha", "mulato",
    #"linda", "lindo", "loira", "loiro", "morena",
    "gata", "gatinha", "gato", "musa", "sereia", "cigana", "cigano"
}

_NLP = None

# ============================================================
# NORMALIZAÇÃO
# ============================================================

def carregar_spacy():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy não está instalado. Instale com:\n"
            "  pip install spacy\n"
            "  python -m spacy download pt_core_news_sm"
        ) from exc
    try:
        _NLP = spacy.load(MODELO_SPACY, disable=["parser", "ner"])
    except OSError as exc:
        raise RuntimeError(
            f"Modelo spaCy '{MODELO_SPACY}' não encontrado. Instale com:\n"
            f"  python -m spacy download {MODELO_SPACY}"
        ) from exc
    return _NLP


def limpar_texto_para_spacy(texto: str) -> str:
    texto = str(texto).lower()
    texto = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s]", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def remover_acentos(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def normalizar_alvo(texto: str) -> str:
    """
    Normalização dos ALVOS.
    Preserva acentos para não misturar pares como avó/avô e vovó/vovô.
    Não usa lema.
    """
    texto = str(texto).lower().strip()
    texto = re.sub(r"[^a-zà-öø-ÿ0-9]", "", texto)
    return texto


def normalizar_atributo(texto: str) -> str:
    """
    Normalização dos ATRIBUTOS.
    Remove acentos porque, para atributos, queremos aproximar saúde/saude,
    paixão/paixao etc. A distinção avó/avô não se aplica aqui.
    """
    texto = remover_acentos(texto)
    texto = re.sub(r"[^a-z0-9]", "", texto)
    return texto.strip()


def tokenizar_com_spacy(texto: str) -> list[dict]:
    """
    Retorna tokens com duas normalizações:
    - alvo_norm: para ALVOS, sem lematização e com acentos preservados;
    - attr_raw_norm/attr_lemma_norm: para ATRIBUTOS.
    """
    texto_limpo = limpar_texto_para_spacy(texto)
    if not texto_limpo:
        return []
    nlp = carregar_spacy()
    doc = nlp(texto_limpo)
    saida = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        raw = tok.text.lower()
        lema = tok.lemma_.lower() if tok.lemma_ else raw
        alvo_norm = normalizar_alvo(raw)
        attr_raw_norm = normalizar_atributo(raw)
        attr_lemma_norm = normalizar_atributo(lema)
        if alvo_norm or attr_raw_norm or attr_lemma_norm:
            saida.append({
                "raw": raw,
                "lemma": lema,
                "alvo_norm": alvo_norm,
                "attr_raw_norm": attr_raw_norm,
                "attr_lemma_norm": attr_lemma_norm,
            })
    return saida


def lematizar_palavra_atributo(palavra: str) -> str:
    pares = tokenizar_com_spacy(palavra)
    if not pares:
        return normalizar_atributo(palavra)
    return pares[0]["attr_lemma_norm"] or pares[0]["attr_raw_norm"]

# ============================================================
# PREPARAÇÃO DAS LISTAS
# ============================================================

def preparar_alvos() -> dict[str, set[str]]:
    """
    ALVOS SEM LEMATIZAÇÃO.
    A lista já contém singular, plural, diminutivo e aumentativo.
    """
    resultado = {}
    for genero, palavras in ALVOS.items():
        formas = set()
        for palavra in palavras:
            forma = normalizar_alvo(palavra)
            if forma:
                formas.add(forma)
        resultado[genero] = formas
    return resultado


def preparar_atributos() -> dict[str, dict[str, set[str]]]:
    """
    ATRIBUTOS com lematização segura.
    Atributos problemáticos ficam com correspondência por forma original normalizada.
    """
    resultado = {}
    for categoria, palavras in ATRIBUTOS.items():
        resultado[categoria] = {}
        for palavra in palavras:
            canonica = normalizar_atributo(palavra)
            variantes = {canonica}
            if canonica not in ATRIBUTOS_SEM_LEMA_AUTOMATICO:
                lema = lematizar_palavra_atributo(palavra)
                if lema:
                    variantes.add(lema)
            resultado[categoria][canonica] = variantes
    return resultado

# ============================================================
# EXTRAÇÃO DE CONTEXTOS
# ============================================================

def extrair_contextos(df: pd.DataFrame, alvos_norm: dict[str, set[str]]) -> list[dict]:
    contextos = []
    for idx_linha, row in df.iterrows():
        tokens = tokenizar_com_spacy(row.get(COLUNA_TEXTO, ""))
        if not tokens:
            continue

        musica = row.get(COLUNA_MUSICA, "")
        artista = row.get(COLUNA_ARTISTA, "")

        for i, token in enumerate(tokens):
            token_alvo = token["alvo_norm"]

            for genero, alvos_genero in alvos_norm.items():
                if token_alvo not in alvos_genero:
                    continue

                ini = max(0, i - JANELA_CONTEXTO)
                fim = min(len(tokens), i + JANELA_CONTEXTO + 1)

                if REMOVER_ALVO_DO_CONTEXTO:
                    tokens_contexto = tokens[ini:i] + tokens[i + 1:fim]
                else:
                    tokens_contexto = tokens[ini:fim]

                if not tokens_contexto:
                    continue

                contextos.append({
                    "idx_linha_csv": idx_linha,
                    "genero": genero,
                    "alvo": token["raw"],
                    "alvo_norm": token_alvo,
                    "alvo_lema_apenas_auditoria": token["lemma"],
                    "contexto": " ".join(t["raw"] for t in tokens_contexto),
                    "contexto_lematizado": " ".join(t["lemma"] for t in tokens_contexto),
                    "contexto_attr_raw": [t["attr_raw_norm"] for t in tokens_contexto if t["attr_raw_norm"]],
                    "contexto_attr_lemas": [t["attr_lemma_norm"] for t in tokens_contexto if t["attr_lemma_norm"]],
                    "musica": musica,
                    "artista": artista,
                })

    if not DEDUPLICAR_CONTEXTOS:
        return contextos

    # Mantido só por segurança; por configuração, não é usado.
    vistos = set()
    unicos = []
    for c in contextos:
        chave = (c["genero"], c["alvo_norm"], c["contexto"], c["musica"], c["artista"])
        if chave in vistos:
            continue
        vistos.add(chave)
        unicos.append(c)
    return unicos

# ============================================================
# CONTAGEM DE ATRIBUTOS
# ============================================================

def contar_variantes_no_contexto(contexto: dict, variantes: set[str]) -> int:
    """
    Conta ocorrências no contexto SEM deduplicar contextos e SEM transformar
    a janela em conjunto.

    Correção importante:
    - cada token da janela conta no máximo 1 vez para a mesma palavra canônica;
    - isso evita duplicação técnica quando token bruto e lema são iguais
      (ex.: gostoso/gostoso, fogo/fogo);
    - repetições reais em tokens diferentes continuam contando.
    """
    total = 0
    raws = contexto.get("contexto_attr_raw", [])
    lemas = contexto.get("contexto_attr_lemas", [])

    for raw, lema in zip(raws, lemas):
        if raw in variantes or lema in variantes:
            total += 1

    return total


def contexto_tem_variantes(contexto: dict, variantes: set[str]) -> bool:
    return contar_variantes_no_contexto(contexto, variantes) > 0


def pmi_suavizada(n_total: int, n_genero: int, n_attr_total: int, n_attr_genero: int) -> float:
    p_y_dado_x = (n_attr_genero + EPSILON) / (n_genero + 2 * EPSILON)
    p_y = (n_attr_total + EPSILON) / (n_total + 2 * EPSILON)
    return math.log2(p_y_dado_x / p_y)


def media_ponderada(valores: pd.Series, pesos: pd.Series) -> float:
    tmp = pd.DataFrame({"valor": valores, "peso": pesos}).dropna()
    if tmp.empty or tmp["peso"].sum() == 0:
        return np.nan
    return float((tmp["valor"] * tmp["peso"]).sum() / tmp["peso"].sum())

# ============================================================
# GRÁFICOS
# ============================================================

def grafico_barras(df: pd.DataFrame, coluna: str, nome: str, titulo: str, ylabel: str):
    pivot = df.pivot(index="categoria", columns="genero", values=coluna).reindex(ORDEM_CATEGORIAS)
    x = np.arange(len(ORDEM_CATEGORIAS))
    largura = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - largura / 2, pivot.get("Masculino", pd.Series(index=ORDEM_CATEGORIAS, dtype=float)), largura, label="Masculino")
    ax.bar(x + largura / 2, pivot.get("Feminino", pd.Series(index=ORDEM_CATEGORIAS, dtype=float)), largura, label="Feminino")
    if "pmi" in coluna:
        ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(titulo)
    ax.set_xlabel("Categoria")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDEM_CATEGORIAS, rotation=45, ha="right")
    ax.legend(title="Gênero")
    plt.tight_layout()
    plt.savefig(nome, dpi=200)
    plt.close()


def grafico_aparencia_top(df_palavras: pd.DataFrame, nome: str):
    tmp = df_palavras[df_palavras["categoria"] == "Aparência"].copy()
    tmp = tmp[tmp["ocorrencias_atributo_genero"] > 0]
    if tmp.empty:
        return
    top_palavras = (tmp.groupby("palavra_canonica")["ocorrencias_atributo_genero"]
                      .sum().sort_values(ascending=False).head(20).index.tolist())
    tmp = tmp[tmp["palavra_canonica"].isin(top_palavras)]
    pivot = tmp.pivot_table(index="palavra_canonica", columns="genero", values="ocorrencias_atributo_genero", aggfunc="sum", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
    ax = pivot.plot(kind="barh", figsize=(10, 7))
    ax.set_title("Top palavras de Aparência por gênero")
    ax.set_xlabel("Ocorrências em contextos de alvo")
    ax.set_ylabel("Palavra")
    plt.tight_layout()
    plt.savefig(nome, dpi=200)
    plt.close()

# ============================================================
# MAIN
# ============================================================

def main():
    caminho = Path(CAMINHO_CSV)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho.resolve()}")

    carregar_spacy()
    df = pd.read_csv(caminho, encoding="utf-8-sig")
    if COLUNA_TEXTO not in df.columns:
        raise ValueError(f"Coluna '{COLUNA_TEXTO}' não encontrada. Colunas: {list(df.columns)}")
    if USAR_PRIMEIRAS_N_MUSICAS is not None:
        df = df.head(USAR_PRIMEIRAS_N_MUSICAS).copy()

    print("Preparando listas...")
    alvos_norm = preparar_alvos()
    atributos_norm = preparar_atributos()

    colisoes = sorted(alvos_norm.get("Feminino", set()) & alvos_norm.get("Masculino", set()))
    pd.DataFrame({"alvo_normalizado_em_ambos_generos": colisoes}).to_csv(
        f"{PREFIXO_SAIDA}_diagnostico_colisoes_alvos.csv", index=False, encoding="utf-8-sig"
    )

    print("Extraindo contextos sem deduplicar...")
    contextos = extrair_contextos(df, alvos_norm)
    if not contextos:
        raise ValueError("Nenhum contexto foi encontrado. Confira a coluna de letras e as listas de alvos.")

    df_contextos = pd.DataFrame([{k: v for k, v in c.items() if k not in ["contexto_attr_raw", "contexto_attr_lemas"]} for c in contextos])
    df_contextos.to_csv(f"{PREFIXO_SAIDA}_contextos.csv", index=False, encoding="utf-8-sig")

    n_total_contextos = len(contextos)
    contextos_por_genero = {g: [c for c in contextos if c["genero"] == g] for g in ALVOS}
    n_genero = {g: len(lst) for g, lst in contextos_por_genero.items()}

    print(f"Músicas analisadas: {len(df)}")
    print(f"Contextos usados: {n_total_contextos}")
    print(f"Contextos por gênero: {n_genero}")
    print(f"Colisões entre alvos normalizados: {len(colisoes)}")
    if colisoes:
        print("ATENÇÃO: há alvos iguais nos dois gêneros após normalização:", colisoes)

    diag_alvos = (df_contextos.groupby(["genero", "alvo_norm", "alvo"], dropna=False)
                  .size().reset_index(name="qtd_contextos")
                  .sort_values(["genero", "qtd_contextos"], ascending=[True, False]))
    diag_alvos.to_csv(f"{PREFIXO_SAIDA}_diagnostico_alvos_encontrados.csv", index=False, encoding="utf-8-sig")

    linhas_palavras = []
    for categoria, palavras in atributos_norm.items():
        print(f"Contando categoria: {categoria}")
        for palavra_canonica, variantes in palavras.items():
            cont_total = sum(contar_variantes_no_contexto(c, variantes) for c in contextos)
            ctx_total = sum(1 for c in contextos if contexto_tem_variantes(c, variantes))
            entra = cont_total >= MIN_OCORRENCIA_ATRIBUTO
            for genero, lista_ctx in contextos_por_genero.items():
                cont_genero = sum(contar_variantes_no_contexto(c, variantes) for c in lista_ctx)
                ctx_genero = sum(1 for c in lista_ctx if contexto_tem_variantes(c, variantes))
                pmi = pmi_suavizada(n_total_contextos, n_genero[genero], ctx_total, ctx_genero) if entra else np.nan
                linhas_palavras.append({
                    "genero": genero,
                    "categoria": categoria,
                    "palavra_canonica": palavra_canonica,
                    "variantes_consideradas": "; ".join(sorted(variantes)),
                    "n_total_contextos": n_total_contextos,
                    "n_contextos_genero": n_genero[genero],
                    "contextos_com_atributo_baseline": ctx_total,
                    "contextos_com_atributo_genero": ctx_genero,
                    "ocorrencias_atributo_baseline": cont_total,
                    "ocorrencias_atributo_genero": cont_genero,
                    "p_contexto_atributo_baseline": ctx_total / n_total_contextos if n_total_contextos else np.nan,
                    "p_contexto_atributo_genero": ctx_genero / n_genero[genero] if n_genero[genero] else np.nan,
                    "ocorrencias_por_100_contextos_genero": (cont_genero / n_genero[genero] * 100) if n_genero[genero] else np.nan,
                    "pmi_por_contexto_binario": pmi,
                    "usada_na_media": entra,
                    "motivo": "ok" if entra else "nao_ocorre_no_baseline",
                })

    df_palavras = pd.DataFrame(linhas_palavras)
    df_palavras.to_csv(f"{PREFIXO_SAIDA}_palavras.csv", index=False, encoding="utf-8-sig")

    linhas_cat = []
    for categoria in ORDEM_CATEGORIAS:
        for genero in ["Masculino", "Feminino"]:
            grupo = df_palavras[(df_palavras["categoria"] == categoria) & (df_palavras["genero"] == genero) & (df_palavras["usada_na_media"])]
            lista_ctx = contextos_por_genero[genero]
            palavras_cat = atributos_norm[categoria]

            ocorrencias_categoria_genero = 0
            contextos_com_categoria_genero = 0
            for c in lista_ctx:
                total_c = sum(contar_variantes_no_contexto(c, variantes) for variantes in palavras_cat.values())
                ocorrencias_categoria_genero += total_c
                if total_c > 0:
                    contextos_com_categoria_genero += 1

            ocorrencias_categoria_total = 0
            contextos_com_categoria_total = 0
            for c in contextos:
                total_c = sum(contar_variantes_no_contexto(c, variantes) for variantes in palavras_cat.values())
                ocorrencias_categoria_total += total_c
                if total_c > 0:
                    contextos_com_categoria_total += 1

            linhas_cat.append({
                "genero": genero,
                "categoria": categoria,
                "n_contextos_genero": n_genero[genero],
                "contextos_com_categoria_genero": contextos_com_categoria_genero,
                "ocorrencias_categoria_genero": ocorrencias_categoria_genero,
                "contextos_com_categoria_por_100": (contextos_com_categoria_genero / n_genero[genero] * 100) if n_genero[genero] else np.nan,
                "ocorrencias_categoria_por_100_contextos": (ocorrencias_categoria_genero / n_genero[genero] * 100) if n_genero[genero] else np.nan,
                "pmi_categoria_contexto_binario": pmi_suavizada(n_total_contextos, n_genero[genero], contextos_com_categoria_total, contextos_com_categoria_genero),
                "pmi_medio_palavras": grupo["pmi_por_contexto_binario"].mean() if not grupo.empty else np.nan,
                "pmi_ponderado_por_ocorrencia": media_ponderada(grupo["pmi_por_contexto_binario"], grupo["ocorrencias_atributo_genero"] + EPSILON) if not grupo.empty else np.nan,
                "qtd_palavras_categoria": len(palavras_cat),
                "qtd_palavras_usadas": int(grupo["palavra_canonica"].nunique()) if not grupo.empty else 0,
            })

    df_cat = pd.DataFrame(linhas_cat)
    df_cat["categoria"] = pd.Categorical(df_cat["categoria"], categories=ORDEM_CATEGORIAS, ordered=True)
    df_cat = df_cat.sort_values(["categoria", "genero"])
    df_cat.to_csv(f"{PREFIXO_SAIDA}_categorias.csv", index=False, encoding="utf-8-sig")

    # Contextos de aparência para auditoria manual
    aparencia = atributos_norm.get("Aparência", {})
    linhas_aparencia = []
    for c in contextos:
        encontradas = []
        for palavra, variantes in aparencia.items():
            qtd = contar_variantes_no_contexto(c, variantes)
            if qtd > 0:
                encontradas.extend([palavra] * qtd)
        if encontradas:
            linhas_aparencia.append({
                "genero": c["genero"],
                "alvo": c["alvo"],
                "alvo_norm": c["alvo_norm"],
                "atributos_aparencia": "; ".join(encontradas),
                "qtd_aparencia": len(encontradas),
                "contexto": c["contexto"],
                "contexto_lematizado": c["contexto_lematizado"],
                "musica": c["musica"],
                "artista": c["artista"],
            })
    pd.DataFrame(linhas_aparencia).to_csv(f"{PREFIXO_SAIDA}_auditoria_contextos_aparencia.csv", index=False, encoding="utf-8-sig")

    auditoria_palavras = df_palavras.sort_values(["categoria", "genero", "ocorrencias_atributo_genero", "pmi_por_contexto_binario"], ascending=[True, True, False, False])
    auditoria_palavras.to_csv(f"{PREFIXO_SAIDA}_auditoria_palavras.csv", index=False, encoding="utf-8-sig")

    # ========================================================
    # 4 GRÁFICOS PRINCIPAIS
    # ========================================================
    grafico_barras(
        df_cat,
        "contextos_com_categoria_por_100",
        f"{PREFIXO_SAIDA}_grafico_01_contextos_com_categoria_por_100.png",
        "Contextos com pelo menos um atributo da categoria por 100 contextos",
        "Contextos por 100",
    )
    grafico_barras(
        df_cat,
        "pmi_medio_palavras",
        f"{PREFIXO_SAIDA}_grafico_02_pmi_medio_palavras.png",
        "PMI médio por atributo",
        "PMI médio",
    )
    grafico_barras(
        df_cat,
        "pmi_ponderado_por_ocorrencia",
        f"{PREFIXO_SAIDA}_grafico_03_pmi_ponderado_por_ocorrencia.png",
        "PMI ponderado por frequência dos atributos",
        "PMI ponderado",
    )
    grafico_aparencia_top(
        df_palavras,
        f"{PREFIXO_SAIDA}_grafico_04_top_aparencia_por_genero.png"
    )

    for coluna, sufixo in [
        ("contextos_com_categoria_por_100", "comparacao_contextos_com_categoria_por_100"),
        ("pmi_medio_palavras", "comparacao_pmi_medio_palavras"),
        ("pmi_ponderado_por_ocorrencia", "comparacao_pmi_ponderado_por_ocorrencia"),
    ]:
        comparacao = df_cat.pivot(index="categoria", columns="genero", values=coluna).reset_index()
        if "Feminino" in comparacao.columns and "Masculino" in comparacao.columns:
            comparacao["diferenca_feminino_masculino"] = comparacao["Feminino"] - comparacao["Masculino"]
        comparacao.to_csv(f"{PREFIXO_SAIDA}_{sufixo}.csv", index=False, encoding="utf-8-sig")

    print("\nResumo por categoria:")
    print(df_cat.to_string(index=False))
    print("\nArquivos gerados:")
    for nome in [
        f"{PREFIXO_SAIDA}_contextos.csv",
        f"{PREFIXO_SAIDA}_palavras.csv",
        f"{PREFIXO_SAIDA}_categorias.csv",
        f"{PREFIXO_SAIDA}_auditoria_contextos_aparencia.csv",
        f"{PREFIXO_SAIDA}_auditoria_palavras.csv",
        f"{PREFIXO_SAIDA}_diagnostico_alvos_encontrados.csv",
        f"{PREFIXO_SAIDA}_diagnostico_colisoes_alvos.csv",
        f"{PREFIXO_SAIDA}_comparacao_contextos_com_categoria_por_100.csv",
        f"{PREFIXO_SAIDA}_comparacao_pmi_medio_palavras.csv",
        f"{PREFIXO_SAIDA}_comparacao_pmi_ponderado_por_ocorrencia.csv",
        f"{PREFIXO_SAIDA}_grafico_01_contextos_com_categoria_por_100.png",
        f"{PREFIXO_SAIDA}_grafico_02_pmi_medio_palavras.png",
        f"{PREFIXO_SAIDA}_grafico_03_pmi_ponderado_por_ocorrencia.png",
        f"{PREFIXO_SAIDA}_grafico_04_top_aparencia_por_genero.png",
    ]:
        print("-", nome)


if __name__ == "__main__":
    main()
