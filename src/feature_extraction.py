"""
Modulo per l'estrazione delle feature ausiliarie simboliche e morfologiche per la
classificazione dei dialetti/lingue regionali italiane.

Blocchi di feature (concatenati, poi standardizzati):
 1. Char n-grams TF-IDF (char_wb, 2-4)            -> fonetica/desinenze subword
 2. Word-edge TF-IDF (prefissi/suffissi 1-3 + parole funzione brevi)
                                                  -> desinenze (-u, -s, -ei...), clitici, articoli
 3. Statistiche ortografiche generali             -> geminate, cluster consonantici,
                                                     vocale finale, apostrofi, diacritici...
 4. Tokenizer fertility (4 valori)                -> fertility media, max per parola,
                                                     % parole molto frammentate, % [UNK]
 5. Marcatori specifici per dialetto (11 dialetti, ~90 regex)
                                                  -> tratti fonetici, grammaticali e lessicali
                                                     + un punteggio aggregato per dialetto

Note di progetto:
 - Tutti i testi sono normalizzati una sola volta (NFC, apostrofi unificati)
   SENZA rimuovere i diacritici (ü, ö, ł, â, ç, ə... sono segnali discriminanti).
   Il testo normalizzato mantiene le maiuscole: la fertility del tokenizer si calcola
   su di esso (il tokenizer è cased), mentre TF-IDF, statistiche ortografiche e regex
   usano la versione in minuscolo (.lower()).
 - Le feature a regex sono "rate" per parola (conteggio / n_parole) per essere
   indipendenti dalla lunghezza della frase.
 - fit/transform lavorano a batch: la matrice densa completa non viene mai
   materializzata in fit (StandardScaler.partial_fit).
 - Dopo la standardizzazione i valori sono clippati: feature rarissime avrebbero
   altrimenti z-score enormi che destabilizzano il ramo ausiliario della rete.
"""

import math
import re
import unicodedata
from collections import Counter

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Normalizzazione e tokenizzazione "per parole"
# --------------------------------------------------------------------------- #
_APOS_TABLE = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'", "′": "'"})
_L = r"[^\W\d_]"  # una lettera unicode
# parola = lettere con apostrofi interni (s'èt, gh'ò, l'è, n'agghia)
_WORD_RE = re.compile(rf"{_L}+(?:'{_L}+)*")


def normalize_text(text, lower=True):
    """NFC + apostrofi unificati (+ lowercase se lower=True). Mantiene i diacritici."""
    out = unicodedata.normalize("NFC", str(text)).translate(_APOS_TABLE)
    return out.lower() if lower else out


def _edge_analyzer(text):
    """
    Analyzer per il vectorizer 'word-edge' (il testo arriva già normalizzato).
    Emette: prefissi ^x, ^xy, ^xyz; suffissi x$, xy$, xyz$; e le parole brevi intere
    (<=4 lettere: xe, nun, cu, ël, minga...), che sono spesso parole funzione dialettali.
    """
    feats = []
    for w in _WORD_RE.findall(text):
        n = len(w)
        for k in (1, 2, 3):
            if n >= k:
                feats.append("^" + w[:k])
                feats.append(w[-k:] + "$")
        if n <= 4:
            feats.append("=" + w + "=")
    return feats


# --------------------------------------------------------------------------- #
# Marcatori linguistici per dialetto (applicati a testo normalizzato/lowercase)
# --------------------------------------------------------------------------- #
_FINAL_U = rf"\b{_L}{{2,}}u\b"                         # -u finale (siciliano, sardo, salentino)
_LAT_CLUSTERS = r"\b(?:cl|fl|pl|bl|gl)[aeiouâêîôû]"    # conservazione cluster latini iniziali
_PLURAL_S = r"\b\w{3,}[aeiu]s\b"                       # plurali sigmatici (-as, -es, -is, -us)

DIALECT_MARKERS = {
    "piemontese": {
        "front_vowels": r"[üöë]",
        "nen": r"\bnen\b",                              # negazione "non"
        "pa_mach": r"\b(?:pa|mach)\b",
        "massa": r"\bmassa\b",                          # troppo
        "boia_faus": r"\bbo[ij]a\s+fa[uù]s\b",
        "articles": r"\b(?:ël|dël|ij|dij|sël)\b",
        "a_l_e": r"\ba\s+l'[eèé]\b",
        "subj_clitic": r"\b(?:i|it|a)\s+(?:son|ses|soma|sioma|sio|veuj|veus|l'ha|l'han)\b",
    },
    "lombardo": {
        "minga": r"\bminga\b",                          # negazione post-verbale
        "cadrega": r"\bcadreg\w*",                      # sedia
        "tel_chi": r"\btel\s+(?:ch[iì]|l[aà])\b",       # eccolo qui/là
        "pronouns": r"\b(?:mì|tì|lü|lée|lee|nüm|num|nialter|vialter)\b",
        "front_vowels": r"[üö]",
        "scior": r"\bsci[oô]r\w*",
        "fioeu": r"\bfi(?:oe|ö)\w*",
    },
    "emiliano": {
        "dre_a": r"\bdr[eēê]\s+a\b",                    # aspetto progressivo
        "socmel": r"\bsoc'?\s*m[eè]l\b",
        "gh_apos": r"\bgh'[a-zàèéòô]+",                 # a gh'ò
        "s_et": r"\bs'[eè]t\b",
        "subj_clitic": r"\b(?:al|la|i|a)\s+(?:magna|vol|dis|vén|vèn|dorm\w*|pol)\b",
        "circ_macron": r"[êôēō]",
    },
    "genovese": {
        "internal_x": r"(?<=[^\W\d_])x(?=[^\W\d_])",    # x = [ʒ] (coxina)
        "belin": r"\bbel[iì]n\b",
        "caruggi": r"\bcaru[gj]\w*",
        "cluster_shift": r"\bsciumm?e\b|\bchi[aà]n\b",  # fiume, piano
        "scignor": r"\bsci?[gñ]n?or\w*",
        "special_letters": r"[æœçñ]",
        "mugugno": r"\bmugugn\w*",
    },
    "veneto": {
        "xe": r"\bx[eéè]\b",                            # è
        "x_initial": r"\bx\w*",
        "l_bar": r"ł",                                  # L evanescente
        "ghe": r"\bghe\b",
        "ciacole": r"\bci[àa]co[lł]\w*",
        "sior": r"\bsior\w*",
        "ndar": r"\bnd(?:ar|à|em|emo)\b",
        "go": r"\bgo\b",
        "deso": r"\bdeso\b",
    },
    "friulano": {
        "mandi": r"\bmandi\b",
        "cj_gj": r"[cg]j",                              # cjase, gjat
        "circumflex": r"[âêîôû]",                       # vocali lunghe (lât)
        "lat_clusters": _LAT_CLUSTERS,
        "article_les": r"\bles\b",
        "cuan": r"\bcuan\w*",
        "plural_s": _PLURAL_S,
    },
    "ladino": {
        "cia": r"\bcia(?:s|r|f)\w*",                    # ca- -> cia-
        "bun_de": r"\bbun\s+d[eéè]\b",
        "dilan": r"\bdilan\b",
        "front_vowels": r"[ëöü]",
        "el_pron": r"\bël\b",
        "sce": r"\bsce\b",
        "lat_clusters": _LAT_CLUSTERS,
        "plural_s": _PLURAL_S,
    },
    "napoletano": {
        "guaglio": r"\bguagli\w*",
        "mannaggia": r"\bmannagg\w*",
        "quanno_chiummo": r"\bquann\w*|\bchiumm\w*",    # nd->nn, mb->mm
        "cca_lla": r"\bcc[aà]\b|\bll[aà]\b",
        "chisto_chello": r"\bchist[oaie]\b|\bchell[oaie]\b",
        "jamme": r"\bjamm?[oe]\b",
        "nun": r"\bnun\b",
        "articles": r"\b(?:nu|na)\b",
        "lead_apos": r"(?<!\w)'(?=[^\W\d_])",           # 'o, 'a, 'e (aferesi)
        "schwa": r"ə",
        "enclitic_poss": r"\b(?:fr[àa]t|s[òo]r|mamm|pat|figli|marit|mogli|nonn)[eo](?:mo|ma|ta|to|sa|so)\b",
        "cchiu": r"\bc?chi[uù]\b",
        "mo": r"\bmo\b",
    },
    "tarantino": {
        "cu": r"\bcu\b",                                # cu + verbo finito
        "ca": r"\bca\b",
        "ce_ste_face": r"\bc[eé]\s+st[eé]\b|\bst[eé]\s+fac[eè]\b",
        "mo": r"\bmo\b",
        "article_u": r"\bu\b",
        "nun": r"\bnun\b",
        "face_accent": r"\bfac[èé]\b",
        "final_u": _FINAL_U,
    },
    "siciliano": {
        "dd_final": r"dd[ua]\b",                        # beddu, bedda
        "ddr": r"ddr",
        "final_u": _FINAL_U,
        "amuninni": r"\bamuninn?i\b|\bamuni\b",
        "mizzica": r"\bmizzic\w*|\bminchi\w*",
        "aviri_a": r"\b(?:haju|aju|haiu|aiu|ajiu)\s+a\b",
        "a_object": r"\b(?:viju|vidu|canusciu)\s+a\b",
        "iddu": r"\bidd[ua]\b|\biddi\b",
        "nuatri": r"\b[nv]uatri\b",
        "unni": r"\bunni\b",
        "cca_dda": r"\bcc[aà]\b|\bdd[aà]\b",
        "picciriddu": r"\bpiccir\w*|\bpicciott\w*|\bpicchi\b",
        "nun": r"\bnun\b",
        "article_u": r"\bu\b",
        "cchiu": r"\bc?chi[uù]\b",
    },
    "sardo": {
        "plural_articles": r"\b(?:sos|sas)\b",
        "singular_articles": rf"\bs[ua]\s+{_L}{{3,}}",
        "ajo": r"\baj[òó]\b|\bajo\b",
        "eja_nono": r"\b(?:eja|nono)\b",
        "k_letter": r"k",
        "tz": r"tz",
        "pitzinnu": r"\bpitzinn\w*",
        "lexicon": r"\b(?:deo|deu|cun|meda|mannu|manna|bonu|bona|issos|isse|issa)\b",
        "plural_s": _PLURAL_S,
        "final_u": _FINAL_U,
        "lat_clusters": _LAT_CLUSTERS,
    },
}

_DIACR = re.compile(r"[àèéìòùáíóúâêîôûäëïöüÿāēīōūœæçñłə]")
_CIRC = re.compile(r"[âêîôûāēīōū]")
_FRONT = re.compile(r"[üöœëæ]")
_GEM = re.compile(r"([bcdfglmnprstvz])\1")
_CL3 = re.compile(r"[bcdfghjklmnpqrstvwxz]{3,}")
_CL4 = re.compile(r"[bcdfghjklmnpqrstvwxz]{4,}")
_LEAD_APOS = re.compile(r"(?<!\w)'(?=[^\W\d_])")
_CONS = frozenset("bcdfghjklmnpqrstvwxyz")

_COMPILED_MARKERS = {
    d: {n: re.compile(p) for n, p in feats.items()} for d, feats in DIALECT_MARKERS.items()
}

_GENERAL_NAMES = [
    "log_n_words", "mean_word_len", "apos_rate", "lead_apos_rate", "diacritic_rate",
    "circumflex_rate", "front_vowel_rate", "geminate_rate", "cluster3_rate", "cluster4_rate",
    "final_a", "final_e", "final_i", "final_o", "final_u", "final_accented", "final_schwa",
    "final_consonant", "final_s", "x_rate", "k_rate", "j_rate", "l_bar_rate", "tz_rate"
]
_TOKENIZER_NAMES = ["fertility", "max_word_fertility", "frac_words_ge3_pieces", "unk_ratio"]


class DialectAuxiliaryFeatureExtractor:
    def __init__(
        self,
        tokenizer,
        max_char_features=500,
        char_ngram_range=(2, 4),
        max_edge_features=300,
        min_df=5,
        clip_value=10.0,
        batch_size=4096,
        use_dialect_markers=True,
        verbose=True,
    ):
        self.tokenizer = tokenizer
        self.max_char_features = max_char_features
        self.char_ngram_range = char_ngram_range
        self.max_edge_features = max_edge_features
        self.min_df = min_df
        self.clip_value = clip_value
        self.batch_size = batch_size
        self.use_dialect_markers = use_dialect_markers
        self.verbose = verbose

        # n-grammi di caratteri entro i confini di parola (prefissi, radici, desinenze)
        self.char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ngram_range,
            max_features=self.max_char_features,
            sublinear_tf=True,
            min_df=self.min_df,
            lowercase=False,       # il testo è già normalizzato
            strip_accents=None,    # i diacritici sono informativi
            dtype=np.float32,
        )
        # prefissi/suffissi di parola + parole funzione brevi
        self.edge_tfidf = TfidfVectorizer(
            analyzer=_edge_analyzer,
            max_features=self.max_edge_features,
            sublinear_tf=True,
            min_df=self.min_df,
            dtype=np.float32,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

        self._unk = getattr(tokenizer, "unk_token", None)
        self._word_cache = {}
        self._cache_limit = 500_000

        self._marker_names = (
            [f"{d}__{n}" for d, feats in DIALECT_MARKERS.items() for n in feats]
            if use_dialect_markers else []
        )
        self._agg_names = (
            [f"{d}__AGG" for d in DIALECT_MARKERS] if use_dialect_markers else []
        )
        self._hand_names = (
            _GENERAL_NAMES + [f"tok__{n}" for n in _TOKENIZER_NAMES]
            + self._marker_names + self._agg_names
        )
        self.marker_coverage_ = {}

    # ------------------------------------------------------------------ #
    # Utilità
    # ------------------------------------------------------------------ #
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_word_cache"] = {}  # la cache è ricostruibile
        return state

    @staticmethod
    def _normalize_all(texts):
        """NFC + apostrofi unificati, MANTENENDO le maiuscole (il tokenizer è cased).
        Le feature che richiedono minuscolo applicano .lower() al loro interno."""
        return [normalize_text(t, lower=False) for t in texts]

    def _batches(self, items):
        for i in range(0, len(items), self.batch_size):
            yield items[i:i + self.batch_size]

    def _word_stats(self, word):
        """(n_subword, n_unk) per parola, con cache (le parole si ripetono molto)."""
        s = self._word_cache.get(word)
        if s is None:
            toks = self.tokenizer.tokenize(word)
            s = (max(len(toks), 1), sum(1 for t in toks if t == self._unk))
            if len(self._word_cache) < self._cache_limit:
                self._word_cache[word] = s
        return s

    # ------------------------------------------------------------------ #
    # Tokenizer Fertility (+ varianti)
    # ------------------------------------------------------------------ #
    def _tokenizer_stats(self, words):
        """
        fertility = pezzi subword / parole (parole = lettere con apostrofi interni,
        punteggiatura esclusa: così la fertility non è gonfiata da s'èt, gh'ò, l'è).
        Restituisce [fertility, max per parola, % parole con >=3 pezzi, % [UNK]].
        """
        if not words:
            return [1.0, 1.0, 0.0, 0.0]
        stats = [self._word_stats(w) for w in words]
        pieces = [s[0] for s in stats]
        total = sum(pieces)
        return [
            total / len(words),
            float(max(pieces)),
            sum(p >= 3 for p in pieces) / len(words),
            sum(s[1] for s in stats) / max(total, 1),
        ]

    def _compute_fertility(self, texts):
        """Compatibilità con la versione precedente: colonna (N, 1) di fertility."""
        out = [self._tokenizer_stats(_WORD_RE.findall(normalize_text(t, lower=False)))[0] for t in texts]
        return np.array(out, dtype=np.float32).reshape(-1, 1)

    # ------------------------------------------------------------------ #
    # Statistiche ortografiche generali
    # ------------------------------------------------------------------ #
    @staticmethod
    def _general_stats(text, words):
        n = max(len(words), 1)
        n_chars = max(sum(len(w) for w in words), 1)
        finals = Counter(w[-1] for w in words if len(w) >= 2)
        nf = max(sum(finals.values()), 1)
        f_cons = sum(v for k, v in finals.items() if k in _CONS)
        f_acc = sum(v for k, v in finals.items() if k in "àèéìòùáíóú")
        return [
            math.log1p(len(words)),
            n_chars / n,
            text.count("'") / n,
            len(_LEAD_APOS.findall(text)) / n,
            len(_DIACR.findall(text)) / n,
            len(_CIRC.findall(text)) / n,
            len(_FRONT.findall(text)) / n,
            len(_GEM.findall(text)) / n,
            len(_CL3.findall(text)) / n,
            len(_CL4.findall(text)) / n,
            finals["a"] / nf, finals["e"] / nf, finals["i"] / nf, finals["o"] / nf,
            finals["u"] / nf, f_acc / nf, finals["ə"] / nf, f_cons / nf, finals["s"] / nf,
            text.count("x") / n,
            text.count("k") / n,
            text.count("j") / n,
            text.count("ł") / n,
            text.count("tz") / n
        ]

    # ------------------------------------------------------------------ #
    # Marcatori specifici dei dialetti
    # ------------------------------------------------------------------ #
    @staticmethod
    def _marker_stats(text, n_words):
        n = max(n_words, 1)
        rates, aggs = [], []
        for feats in _COMPILED_MARKERS.values():
            total = 0.0
            for pat in feats.values():
                r = len(pat.findall(text)) / n
                rates.append(r)
                total += r
            aggs.append(total)
        return rates + aggs

    def _handcrafted(self, cased_texts):
        """cased_texts: testi normalizzati ma con maiuscole. Fertility sul testo cased,
        statistiche ortografiche e regex sul testo in minuscolo."""
        out = np.zeros((len(cased_texts), len(self._hand_names)), dtype=np.float32)
        for i, cased in enumerate(cased_texts):
            text = cased.lower()
            words = _WORD_RE.findall(text)
            cased_words = _WORD_RE.findall(cased)
            row = self._general_stats(text, words) + self._tokenizer_stats(cased_words)
            if self.use_dialect_markers:
                row += self._marker_stats(text, len(words))
            out[i] = row
        return out

    # ------------------------------------------------------------------ #
    # Assemblaggio
    # ------------------------------------------------------------------ #
    def _assemble(self, norm_batch, hand=None):
        lower_batch = [t.lower() for t in norm_batch]
        char = self.char_tfidf.transform(lower_batch).toarray().astype(np.float32, copy=False)
        edge = self.edge_tfidf.transform(lower_batch).toarray().astype(np.float32, copy=False)
        if hand is None:
            hand = self._handcrafted(norm_batch)
        return np.hstack([char, edge, hand])

    def fit(self, texts):
        """
        Addestra vocabolari TF-IDF e scaler sui testi di training (a batch).
        """
        texts = list(texts)
        if self.verbose:
            print(f"Fit Feature Extractor su {len(texts):,} frasi...")
        norm = self._normalize_all(texts)

        lower_norm = [t.lower() for t in norm]
        self.char_tfidf.fit(lower_norm)
        self.edge_tfidf.fit(lower_norm)
        del lower_norm

        self.scaler = StandardScaler()
        n_dialects = len(DIALECT_MARKERS)
        hits = np.zeros(n_dialects, dtype=np.int64)
        for chunk in self._batches(norm):
            hand = self._handcrafted(chunk)
            if self.use_dialect_markers:
                hits += (hand[:, -n_dialects:] > 0).sum(axis=0)
            self.scaler.partial_fit(self._assemble(chunk, hand))

        if self.use_dialect_markers:
            self.marker_coverage_ = {
                d: float(h) / max(len(norm), 1) for d, h in zip(DIALECT_MARKERS, hits)
            }
        self.is_fitted = True

        if self.verbose:
            nc = len(self.char_tfidf.vocabulary_)
            ne = len(self.edge_tfidf.vocabulary_)
            print(
                f"-> Extractor pronto con {self.n_features_} feature ausiliarie "
                f"(char={nc}, edge={ne}, ortografiche={len(_GENERAL_NAMES)}, "
                f"tokenizer={len(_TOKENIZER_NAMES)}, marcatori={len(self._marker_names)}, "
                f"aggregati={len(self._agg_names)})."
            )
            if self.marker_coverage_:
                cov = ", ".join(f"{d}={c:.1%}" for d, c in self.marker_coverage_.items())
                print(f"   Copertura marcatori (% frasi con >=1 hit): {cov}")
        return self

    @property
    def n_features_(self):
        if not self.is_fitted:
            raise RuntimeError("L'estrattore deve essere addestrato (fit) prima!")
        return (len(self.char_tfidf.vocabulary_) + len(self.edge_tfidf.vocabulary_)
                + len(self._hand_names))

    def get_feature_names_out(self):
        if not self.is_fitted:
            raise RuntimeError("L'estrattore deve essere addestrato (fit) prima!")
        return (
            [f"char:{f}" for f in self.char_tfidf.get_feature_names_out()]
            + [f"edge:{f}" for f in self.edge_tfidf.get_feature_names_out()]
            + self._hand_names
        )

    def transform(self, texts):
        """
        Estrae il tensore PyTorch (N, n_features_) di feature ausiliarie normalizzate.
        """
        if not self.is_fitted:
            raise RuntimeError("L'estrattore deve essere addestrato (fit) prima di invocare transform!")
        norm = self._normalize_all(texts)
        parts = []
        for chunk in self._batches(norm):
            scaled = self.scaler.transform(self._assemble(chunk))
            np.clip(scaled, -self.clip_value, self.clip_value, out=scaled)
            parts.append(scaled.astype(np.float32, copy=False))
        if not parts:
            return torch.empty((0, self.n_features_), dtype=torch.float32)
        return torch.from_numpy(np.vstack(parts))

    def fit_transform(self, texts):
        texts = list(texts)
        return self.fit(texts).transform(texts)