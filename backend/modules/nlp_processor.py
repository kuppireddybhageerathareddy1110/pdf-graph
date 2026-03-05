"""
modules/nlp_processor.py
========================
Extracts named entities and rich relations from text using spaCy.

Relation extraction strategy (4 layers, most→least strict):
  1. SVO triples from dependency parse (subject → verb → object)
  2. Noun-chunk co-occurrence within same sentence
  3. Entity-to-entity pairs in same sentence (cross-type)
  4. Keyword/noun proximity windows across the full page

This multi-layer approach ensures dense graph connectivity even on
technical or terse text where spaCy finds few dependency relations.
"""

from __future__ import annotations
import logging
import re
from collections import defaultdict
from itertools import combinations
from typing import List, Tuple

logger = logging.getLogger(__name__)

_model_cache: dict = {}


def _load_model(model_name: str):
    if model_name not in _model_cache:
        try:
            import spacy
            _model_cache[model_name] = spacy.load(model_name)
            logger.info("Loaded spaCy model: %s", model_name)
        except OSError:
            logger.warning("Model '%s' not found, falling back to en_core_web_sm", model_name)
            import spacy
            _model_cache[model_name] = spacy.load("en_core_web_sm")
    return _model_cache[model_name]


def extract_entities_relations(
    text: str,
    model_name: str = "en_core_web_sm",
    max_entities: int = 150,
    allowed_entity_types: List[str] | None = None,
    window_size: int = 5,
    include_noun_chunks: bool = True,
    include_keyword_nodes: bool = True,
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Extract entities and relations using 4 complementary strategies.

    Parameters
    ----------
    text : str
    model_name : str
    max_entities : int         Hard cap on node count.
    allowed_entity_types : list, optional
    window_size : int          Token window for proximity-based edges.
    include_noun_chunks : bool Add noun chunks as nodes (boosts connectivity).
    include_keyword_nodes: bool Add important nouns/proper nouns as nodes.

    Returns
    -------
    entities  : List[str]
    relations : List[Tuple[str, str, str]]
    """
    if not text or not text.strip():
        return [], []

    nlp = _load_model(model_name)

    if len(text) > 100_000:
        text = text[:100_000]

    doc = nlp(text)

    # ═══════════════════════════════════════════════════════════
    # LAYER 1 — Named entity nodes
    # ═══════════════════════════════════════════════════════════
    seen: set = set()
    entities: List[str] = []

    def _add_entity(name: str) -> bool:
        """Add entity if not duplicate. Returns True if added."""
        name = name.strip()
        key = name.lower()
        if not name or len(name) < 2 or key in seen:
            return False
        if len(entities) >= max_entities:
            return False
        seen.add(key)
        entities.append(name)
        return True

    for ent in doc.ents:
        if allowed_entity_types and ent.label_ not in allowed_entity_types:
            continue
        _add_entity(ent.text)

    # ═══════════════════════════════════════════════════════════
    # LAYER 2 — Noun chunks as nodes (major connectivity booster)
    # ═══════════════════════════════════════════════════════════
    if include_noun_chunks:
        for chunk in doc.noun_chunks:
            text_clean = _clean_chunk(chunk.text)
            if text_clean and len(text_clean.split()) <= 4:
                _add_entity(text_clean)

    # ═══════════════════════════════════════════════════════════
    # LAYER 3 — Important standalone nouns & proper nouns
    # ═══════════════════════════════════════════════════════════
    if include_keyword_nodes:
        for token in doc:
            if (
                token.pos_ in {"NOUN", "PROPN"}
                and not token.is_stop
                and not token.is_punct
                and len(token.text) >= 3
            ):
                _add_entity(token.text)

    # ═══════════════════════════════════════════════════════════
    # RELATION EXTRACTION
    # ═══════════════════════════════════════════════════════════
    relations: List[Tuple[str, str, str]] = []
    seen_edges: set = set()

    def _add_rel(src: str, rel: str, tgt: str):
        src, tgt = src.strip(), tgt.strip()
        if not src or not tgt or src == tgt:
            return
        key = (src.lower(), tgt.lower())
        if key in seen_edges:
            return
        seen_edges.add(key)
        relations.append((src, rel, tgt))

    # ── R1: SVO triples from dependency parse ──────────────────
    for token in doc:
        if token.pos_ != "VERB":
            continue
        verb = token.lemma_

        # Widen subject search: direct children + conjuncts
        subj_tokens = [
            w for w in token.subtree
            if w.dep_ in {"nsubj", "nsubjpass", "csubj", "agent"}
        ]
        obj_tokens = [
            w for w in token.subtree
            if w.dep_ in {"dobj", "attr", "pobj", "iobj",
                          "oprd", "ccomp", "xcomp", "acomp"}
        ]

        for s in subj_tokens:
            for o in obj_tokens:
                _add_rel(s.text, verb, o.text)

        # Also link verb's noun chunk head to its object chunks
        head_chunk = _get_chunk_for_token(token, doc)
        for o in obj_tokens:
            obj_chunk = _get_chunk_for_token(o, doc)
            if head_chunk and obj_chunk and head_chunk != obj_chunk:
                _add_rel(head_chunk, verb, obj_chunk)

    # ── R2: Sentence-level entity co-occurrence ────────────────
    for sent in doc.sents:
        sent_ents = [e.text.strip() for e in sent.ents if e.text.strip() in seen]
        sent_chunks = [
            _clean_chunk(c.text) for c in sent.noun_chunks
            if _clean_chunk(c.text) in seen
        ]
        all_nodes = list(dict.fromkeys(sent_ents + sent_chunks))  # preserve order, dedup

        for a, b in combinations(all_nodes[:20], 2):    # cap at 20 to avoid O(n²) explosion
            _add_rel(a, "related-to", b)

    # ── R3: Token proximity window ─────────────────────────────
    # Slide a window of `window_size` tokens; connect any two entity/chunk
    # nodes that appear within the window.
    tokens = [t for t in doc if not t.is_space]
    entity_set = set(e.lower() for e in entities)

    for i, tok in enumerate(tokens):
        if tok.text.lower() not in entity_set:
            continue
        window_end = min(i + window_size + 1, len(tokens))
        for j in range(i + 1, window_end):
            other = tokens[j]
            if other.text.lower() in entity_set and other.text != tok.text:
                _add_rel(tok.text, "near", other.text)

    # ── R4: Cross-sentence same-entity bridging ────────────────
    # If the same entity appears in multiple sentences, link it to entities
    # in neighboring sentences (bridges disconnected components).
    entity_to_sents: dict = defaultdict(list)
    for i, sent in enumerate(doc.sents):
        for ent in sent.ents:
            name = ent.text.strip()
            if name.lower() in entity_set:
                entity_to_sents[name.lower()].append((i, name))

    for ent_key, appearances in entity_to_sents.items():
        if len(appearances) < 2:
            continue
        # Link consecutive appearances
        for k in range(len(appearances) - 1):
            sent_i, name_i = appearances[k]
            sent_j, name_j = appearances[k + 1]
            if sent_j - sent_i <= 3:   # only bridge nearby sentences
                _add_rel(name_i, "continues-in", name_j)

    logger.debug(
        "Extracted %d entities, %d relations from %d chars",
        len(entities), len(relations), len(text),
    )
    return entities, relations


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_chunk(text: str) -> str:
    """Strip determiners and whitespace from noun chunks."""
    text = re.sub(r"^(the|a|an|this|that|these|those|its|their|our|my|his|her)\s+",
                  "", text.strip(), flags=re.IGNORECASE)
    return text.strip()


def _get_chunk_for_token(token, doc) -> str | None:
    """Return the noun chunk root text that contains this token, or None."""
    for chunk in doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            return _clean_chunk(chunk.text)
    return None
