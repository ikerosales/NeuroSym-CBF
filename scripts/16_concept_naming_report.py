#!/usr/bin/env python
"""Tables and summary of the concept-naming experiment (scripts/15_concept_naming.py).

Reads the already-generated JSON (recomputes nothing) and renders a bilingual (ES/EN, with a
language toggle) HTML with: the methodology explained in plain language, the specificity
distribution of the 128 atoms of NeuroSym-CBF's SAE dictionary, the clearest examples at each
extreme (very specific atoms vs. generic "always active" atoms), and the deep dive into ten
well-known macro tags (scripts/17_concept_tag_deep_dive.py, optional).

Usage:
    python scripts/16_concept_naming_report.py \\
        --input results/concepts/concept_naming_neurosym-cbf-kaggle_ctx30.json
"""
from __future__ import annotations

import argparse
import html
import json
import logging
from pathlib import Path

from fred_forecast.config import data_dir, load_yaml_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CSS = """
  :root {
    --paper:#F4F5F7; --card:#FFFFFF; --ink:#191C22; --ink-mid:#454B58; --muted:#767D8C;
    --rule:#DBDEE4; --accent:#3A4B8F; --accent-w:#E4E7F2; --neg:#A33A3A; --neg-w:#F4E3E3;
    --good:#1F6B4A; --good-w:#DDEDE4;
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,ui-serif,serif;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono:ui-monospace,"SF Mono","Cascadia Mono",Consolas,Menlo,monospace;
  }
  @media (prefers-color-scheme: dark){:root{
    --paper:#14161A;--card:#1C1F26;--ink:#E7EAF0;--ink-mid:#B4BBC8;--muted:#838B9A;
    --rule:#2E333D;--accent:#93A5E8;--accent-w:#232941;--neg:#E08A8A;--neg-w:#332223;
    --good:#6FC49B;--good-w:#1B2E26;}}
  :root[data-theme="dark"]{--paper:#14161A;--card:#1C1F26;--ink:#E7EAF0;--ink-mid:#B4BBC8;
    --muted:#838B9A;--rule:#2E333D;--accent:#93A5E8;--accent-w:#232941;--neg:#E08A8A;
    --neg-w:#332223;--good:#6FC49B;--good-w:#1B2E26;}
  :root[data-theme="light"]{--paper:#F4F5F7;--card:#FFFFFF;--ink:#191C22;--ink-mid:#454B58;
    --muted:#767D8C;--rule:#DBDEE4;--accent:#3A4B8F;--accent-w:#E4E7F2;--neg:#A33A3A;
    --neg-w:#F4E3E3;--good:#1F6B4A;--good-w:#DDEDE4;}
  body{background:var(--paper);color:var(--ink);font-family:var(--serif);font-size:17px;
       line-height:1.62;-webkit-font-smoothing:antialiased}
  .wrap{max-width:46rem;margin:0 auto;padding:2.2rem 1.5rem 5rem;display:flex;
        flex-direction:column;gap:2.6rem}
  header{display:flex;flex-direction:column;gap:.6rem}
  .kicker{font-family:var(--sans);font-size:.74rem;letter-spacing:.13em;text-transform:uppercase;
          color:var(--muted);font-weight:600}
  h1{font-size:clamp(1.9rem,5vw,2.6rem);line-height:1.12;font-weight:600;text-wrap:balance;
     letter-spacing:-.01em}
  .standfirst{color:var(--ink-mid);font-size:1.07rem}
  section{display:flex;flex-direction:column;gap:1rem}
  h2{font-size:1.42rem;font-weight:600;text-wrap:balance;letter-spacing:-.01em;padding-top:.8rem;
     border-top:2px solid var(--ink)}
  h3{font-family:var(--sans);font-size:.96rem;font-weight:650;margin-top:.3rem}
  p{max-width:40rem}
  .scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:4px;background:var(--card)}
  table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:.86rem}
  th,td{padding:.58rem .8rem;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}
  th{font-size:.69rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:650}
  tbody tr:last-child td{border-bottom:none}
  td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}
  tr.hi td{background:var(--accent-w)} tr.hi td:first-child{font-weight:650}
  .better{color:var(--good)} .worse{color:var(--neg)}
  .key{background:var(--good-w);border-left:4px solid var(--good);padding:1.1rem 1.3rem;border-radius:0 4px 4px 0}
  .key p{margin:0}
  .note{font-family:var(--sans);font-size:.85rem;color:var(--ink-mid);background:var(--card);
        border:1px solid var(--rule);border-radius:4px;padding:.6rem .85rem;max-width:40rem}
  dl.defs{display:grid;grid-template-columns:max-content 1fr;gap:.5rem 1.2rem;font-family:var(--sans);
          font-size:.86rem;max-width:40rem}
  dl.defs dt{font-weight:650;white-space:nowrap;font-family:var(--mono)}
  dl.defs dd{margin:0;color:var(--ink-mid)}
  .langbar{position:sticky;top:0;z-index:10;display:flex;justify-content:flex-end;padding:.6rem 0;
           background:var(--paper)}
  .langtoggle{display:flex;gap:.35rem;font-family:var(--sans)}
  .langtoggle button{font:inherit;font-size:.72rem;font-weight:650;letter-spacing:.06em;
      text-transform:uppercase;padding:.35rem .85rem;border-radius:999px;border:1px solid var(--rule);
      background:var(--card);color:var(--muted);cursor:pointer}
  .langtoggle button.active{background:var(--ink);color:var(--paper);border-color:var(--ink)}
  .langblock{display:flex;flex-direction:column;gap:2.6rem}
  body[data-lang="es"] .langblock[data-lang="en"]{display:none}
  body[data-lang="en"] .langblock[data-lang="es"]{display:none}
  @media print{body{background:#fff;font-size:11pt}.wrap{max-width:none;padding:0;gap:1.6rem}
    h2{page-break-after:avoid}.scroll{page-break-inside:avoid}.langbar{display:none}}
"""

_BACKGROUND_ATOM_THRESHOLD = 100_000  # n_active above this: a "background" atom, not a concept one


def strings(lang: str) -> dict:
    if lang == "es":
        return {
            "title": "Nombrado de conceptos: diccionario del SAE",
            "kicker": "NeuroSym-CBF",
            "h1": "¿Qué codifica cada átomo del diccionario?",
            "intro1": (
                "El Sparse Autoencoder Top-K de NeuroSym-CBF aprende un diccionario de "
                "{dict_size} átomos (dict_mult 2 sobre d_model 64, top_k 16 activos por serie). "
                "Este experimento le pone nombre a cada átomo por enriquecimiento de tags: para "
                "cada serie donde un átomo está activo, se comprueba qué tags de FRED aparecen "
                "más a menudo de lo esperable en el corpus completo, con un test hipergeométrico "
                "de una cola. Un lift alto significa que el tag es mucho más común entre las "
                "series que activan ese átomo que entre las series en general, así que es "
                "probable que el átomo esté codificando algo cercano a ese concepto."
            ),
            "intro2": (
                "Ejecutado una vez, sobre el checkpoint de Kaggle (ctx30, el que respalda todas "
                "las cifras de la tesis), sobre el corpus in-sample completo: {n_series} series, "
                "{n_with_tags} de ellas con tags ({pct_tags:.0f}% de cobertura)."
            ),
            "method_h2": "Metodología y qué significa cada número",
            "method_p1": (
                "<strong>¿Qué es un átomo?</strong> NeuroSym-CBF resume la serie en un vector "
                "interno de 128 números (el \"diccionario\" del SAE), de los cuales solo 16 "
                "pueden ser distintos de cero para cada serie (\"top-k\", k=16). Cuando la "
                "posición número <em>i</em> de ese vector es distinta de cero para una serie, "
                "decimos que el \"átomo <em>i</em>\" está <em>activo</em> para esa serie."
            ),
            "method_p2": (
                "<strong>¿Cómo se le pone nombre a un átomo?</strong> Para cada átomo, se mira "
                "el conjunto de series que lo activan y se compara, tag a tag de FRED, qué tan "
                "frecuente es ese tag dentro de ese conjunto frente a en el corpus general. Si un "
                "tag aparece mucho más a menudo dentro del grupo activo, ese tag es un nombre "
                "candidato razonable para el átomo: la misma idea que \"concept naming\" en la "
                "literatura de autoencoders dispersos para LLMs, aquí aplicada a tags económicos "
                "en vez de tokens de texto."
            ),
            "defs_h3": "Qué significa cada columna",
            "defs": [
                ("n_active", "Número de series (de las 194.442 evaluadas) donde el átomo está activo."),
                ("base_rate", "Qué fracción de TODAS las series lleva ese tag. P.ej., base_rate=5.6% para \"unemployment\" significa que el 5,6% de las 194.442 series tiene ese tag, active o no el átomo."),
                ("active_rate", "Qué fracción de las series QUE ACTIVAN el átomo lleva ese tag."),
                ("lift", "active_rate dividido por base_rate. Lift=1 significa que el tag es igual de común dentro y fuera del grupo activo (el átomo no dice nada de ese tag). Lift=10 significa que el tag es 10 veces más común dentro del grupo activo: el átomo sí está relacionado con él."),
                ("recall", "Qué fracción de TODAS las series con ese tag caen dentro del grupo activo del átomo. Lift alto + recall bajo = muy específico pero cubre poco. Recall alto + lift bajo = cubre casi todo, pero no es específico de ese tag (atomo genérico)."),
                ("p-valor", "Probabilidad de ver una coincidencia igual de fuerte (o más) por puro azar, según un test hipergeométrico de una cola. Con decenas de miles de series el p-valor se satura a ~0 incluso para asociaciones débiles, así que sirve para descartar el azar, no para medir especificidad: para eso está el lift."),
                ("átomo de fondo", f"Un átomo activo en más de {_BACKGROUND_ATOM_THRESHOLD:,} de las 194.442 series (más de la mitad del corpus). No codifica un concepto: es estructura compartida por casi toda la base. Se excluye de las tablas de \"qué átomo representa este tag\" porque arrastraría un recall alto para cualquier tag sin decir nada específico."),
            ],
            "bucket_h2": "Cuán específico es el diccionario",
            "bucket_p": (
                "{n_dead} de {dict_size} átomos nunca se activan en ninguna serie. De los "
                "{n_alive} que sí, {n_named} tienen al menos un tag candidato, y {n_selective} "
                "tienen un lift de 5 o más, es decir, un tag al menos 5 veces más común entre las "
                "series que activan ese átomo que en el corpus general: los átomos que se pueden "
                "llamar \"nombrados\" con alguna confianza, no solo \"casi siempre activos\"."
            ),
            "bucket_headers": ["Rango", "Átomos", "% del diccionario"],
            "bucket_labels": [
                "Muerto (nunca se activa)",
                "Sin tag candidato",
                "Lift 1 a 5 (poco específico)",
                "Lift 5 a 20",
                "Lift 20 a 100",
                "Lift 100+ (muy específico)",
            ],
            "specific_h2": "Los átomos más específicos",
            "specific_p": (
                "Ordenados por lift, no por p-valor: con decenas de miles de series el p-valor "
                "se satura a casi cero incluso para un lift de 1,04, así que no separa un átomo "
                "afilado de uno genérico. El lift sí es la señal de cuán específico es un átomo."
            ),
            "specific_headers": ["Atom", "Series activas", "Tag", "Tag expandido", "Lift", "Tasa activa vs base"],
            "no_tag": "(sin tag candidato)",
            "generic_h2": "Los átomos más genéricos",
            "generic_p": (
                "Los átomos con más series activas (hasta {max_active} de {n_series}) tienen "
                "todos un lift cercano a 1: se activan para casi todo, así que ningún tag los "
                "distingue. No están \"sin nombrar\" por falta de intentarlo: son átomos "
                "estructurales (compartidos por casi toda la base), no átomos de concepto."
            ),
            "generic_headers": ["Atom", "Series activas", "Tag más cercano", "Tag expandido", "Lift", "Tasa activa vs base"],
            "deepdive_h2": "Tags macro conocidos",
            "deepdive_p1": (
                "Las tablas de arriba ordenan cada átomo por su único mejor tag, encontrado entre "
                "los 5 candidatos que se guardaron por átomo. Un tag común como \"usa\" o "
                "\"unemployment\" puede perder ese hueco frente a un tag más raro y más "
                "sorprendente incluso en un átomo con el que sí está relacionado, así que esta "
                "sección fija diez tags conocidos y comprueba directamente los 128 átomos para "
                "cada uno, reportando tanto lift (cuánto se enriquece el tag dentro del grupo "
                "activo) como recall (qué fracción de todas las series con ese tag cubre el "
                "átomo de verdad)."
            ),
            "deepdive_p2": (
                "Los diez tags están mejor cubiertos, por recall, por los mismos tres o cuatro "
                "átomos de fondo ya listados arriba (92 a 99% de recall, lift cercano a 1): se "
                "activan en casi toda serie, así que trivialmente cubren casi todo tag también, "
                "sin codificar nada sobre él. Las tablas de abajo excluyen esos átomos de fondo y "
                "muestran los que de verdad son específicos de cada tag."
            ),
            "deepdive_categories_h3": "Las cuatro categorías del veredicto",
            "deepdive_categories": [
                ("Concentrado", "recall ≥ 50% y lift ≥ 10: un único átomo domina y es claramente específico. Solo así lo llamamos \"el átomo de este concepto\"."),
                ("Amplio, no concentrado", "recall ≥ 50% pero lift < 10: el átomo se activa en casi todas las series del tag, pero también en muchísimas otras que no lo llevan. Es un concepto más ancho compartido, no ese tag exacto."),
                ("Fragmentado", "ningún átomo llega a 50% de recall, pero sí hay alguno con lift ≥ 5: el concepto existe en el diccionario, repartido entre varios átomos."),
                ("Ausente", "ni siquiera eso: ningún átomo, fuera de los de fondo, tiene lift ≥ 5 para este tag."),
            ],
            "series_word": "series",
            "of_corpus": "del corpus",
            "cited_note": "El átomo resaltado es el citado en el veredicto de arriba.",
            "no_atoms": "Ningún átomo con recall o lift útil fuera de los de fondo.",
            "deepdive_headers": ["Atom", "Series activas", "Recall", "Lift"],
            "verdicts": {
                "concentrado": "concentrado: el átomo {atom} por sí solo cubre el {recall:.0f}% de las series con este tag, con lift {lift:.1f}",
                "amplio": (
                    "amplio, no concentrado: el átomo {atom} cubre el {recall:.0f}% de las series "
                    "con este tag, pero su lift ({lift:.1f}) es bajo, así que también se activa en "
                    "muchas series que no llevan este tag: es un concepto más ancho compartido, no "
                    "este tag en concreto"
                ),
                "fragmentado": (
                    "fragmentado: el átomo más informativo ({atom}, lift {lift:.1f}) solo cubre un "
                    "{recall:.0f}% de las series con este tag, el resto se reparte entre otros átomos"
                ),
                "ausente": "ausente: ningún átomo, fuera de los de fondo, tiene un lift de 5 o más para este tag",
            },
            "reading_h2": "Lectura",
            "reading1": (
                "El diccionario se reparte en dos grupos claramente distintos: un núcleo pequeño "
                "de átomos muy raros y muy específicos (activos en decenas o cientos de series, "
                "lift de cientos o miles: \"growth\", \"food stamps\", \"chile\", índices "
                "concretos) y un puñado de átomos casi siempre activos (100k+ series, lift ~1) que no "
                "corresponden a ningún tag concreto, sino a estructura compartida por casi toda "
                "la base. Entre esos dos extremos, {n_selective} de {dict_size} átomos "
                "({pct_selective:.0f}%) tienen un tag con lift 5 o más: son la parte del "
                "diccionario que sí se puede nombrar con confianza como \"codifica tal "
                "concepto\", ni tan rara que no generaliza ni tan genérica que no dice nada."
            ),
            "reading2": (
                "Con tags macro conocidos, el patrón se matiza y aparecen cuatro grupos, no dos. "
                "GDP es el único caso limpio (concentrado de verdad): un solo átomo (75) cubre el "
                "91% de sus 152 series con un lift de 121, un concepto raro y coherente que el "
                "SAE sí aprende como unidad propia. Un segundo grupo, amplio pero no concentrado "
                "(cpi, inflation, price index, trade, housing) tiene un átomo con recall alto (69 "
                "a 95%) pero lift apenas de 2 a 3: se activa en la mayoría de las series de ese "
                "tag, pero también en muchísimas otras que no lo llevan, así que es un concepto "
                "más ancho (\"precios e intercambio\") que ese tag exacto. Unemployment y "
                "manufacturing son fragmentados: sí hay un átomo con lift alto (17,3 y 5,9), pero "
                "cubre poco del tag (33% y 2%), señal de que el concepto existe en el diccionario "
                "pero repartido entre varios átomos, no de que el SAE no lo distinga. Y usa y "
                "employment están ausentes de verdad: ningún átomo, dentro o fuera de los de "
                "fondo, tiene un lift útil para ellos, demasiado anchos (employment) o demasiado "
                "ubicuos (usa, 88% del corpus) para que nada se especialice en ellos."
            ),
        }
    return {
        "title": "Concept naming: SAE dictionary",
        "kicker": "NeuroSym-CBF",
        "h1": "What does each atom of the dictionary encode?",
        "intro1": (
            "NeuroSym-CBF's Top-K Sparse Autoencoder learns a dictionary of {dict_size} atoms "
            "(dict_mult 2 over d_model 64, top_k 16 active per series). This experiment names "
            "each atom by tag enrichment: for every series where an atom is active, check which "
            "FRED tags show up more often than their base rate across the full corpus, with a "
            "one-sided hypergeometric test. A high lift means the tag is much more common among "
            "the series that activate that atom than among series in general, so the atom is "
            "likely encoding something close to that tag's concept."
        ),
        "intro2": (
            "Run once, on the Kaggle checkpoint (ctx30, the one behind every number in the "
            "thesis), over the full in-sample corpus: {n_series} series, {n_with_tags} of them "
            "with tags ({pct_tags:.0f}% coverage)."
        ),
        "method_h2": "Methodology and what each number means",
        "method_p1": (
            "<strong>What is an atom?</strong> NeuroSym-CBF summarises the series into an "
            "internal vector of 128 numbers (the SAE's \"dictionary\"), of which only 16 can be "
            "non-zero for any given series (\"top-k\", k=16). When position <em>i</em> of that "
            "vector is non-zero for a series, we say \"atom <em>i</em>\" is <em>active</em> for "
            "that series."
        ),
        "method_p2": (
            "<strong>How is an atom named?</strong> For each atom, look at the set of series "
            "that activate it, and compare, FRED tag by tag, how frequent that tag is inside "
            "that set versus in the general corpus. If a tag shows up much more often inside the "
            "active group, that tag is a reasonable name candidate for the atom, the same idea "
            "as \"concept naming\" in the sparse-autoencoder-for-LLMs literature, applied here to "
            "economic tags instead of text tokens."
        ),
        "defs_h3": "What each column means",
        "defs": [
            ("n_active", "Number of series (out of the 194,442 evaluated) where the atom is active."),
            ("base_rate", "Share of ALL series that carry this tag. E.g., base_rate=5.6% for \"unemployment\" means 5.6% of the 194,442 series have that tag, whether or not the atom is active."),
            ("active_rate", "Share of the series that ACTIVATE the atom which carry this tag."),
            ("lift", "active_rate divided by base_rate. Lift=1 means the tag is equally common inside and outside the active group (the atom says nothing about it). Lift=10 means the tag is 10 times more common inside the active group: the atom is genuinely related to it."),
            ("recall", "Share of ALL series with this tag that fall inside the atom's active group. High lift + low recall = very specific but covers little. High recall + low lift = covers almost everything, but is not specific to this tag (a generic atom)."),
            ("p-value", "Probability of seeing a match this strong (or stronger) by pure chance, from a one-sided hypergeometric test. With tens of thousands of series the p-value saturates to ~0 even for weak associations, so it is useful to rule out chance, not to measure specificity: that is what lift is for."),
            ("background atom", f"An atom active on more than {_BACKGROUND_ATOM_THRESHOLD:,} of the 194,442 series (over half the corpus). It does not encode a concept: it is structure shared by almost the whole base. Excluded from the \"which atom represents this tag\" tables because it would trivially carry a high recall for any tag without meaning anything specific."),
        ],
        "bucket_h2": "How specific is the dictionary",
        "bucket_p": (
            "{n_dead} of {dict_size} atoms never activate on any series. Of the {n_alive} that "
            "do, {n_named} have at least one candidate tag, and {n_selective} have a lift of 5 "
            "or higher, meaning a tag that is at least 5 times more common among the series that "
            "activate that atom than in the corpus overall: the atoms worth calling \"named\" "
            "with any confidence, not just \"almost always active\"."
        ),
        "bucket_headers": ["Range", "Atoms", "% of dictionary"],
        "bucket_labels": [
            "Dead (never activates)",
            "No candidate tag",
            "Lift 1 to 5 (not very specific)",
            "Lift 5 to 20",
            "Lift 20 to 100",
            "Lift 100+ (very specific)",
        ],
        "specific_h2": "The most specific atoms",
        "specific_p": (
            "Ranked by lift, not by p-value: with tens of thousands of series the p-value "
            "saturates to near zero even for a lift of 1.04, so it cannot separate a sharp atom "
            "from a generic one. Lift is the signal of how specific an atom actually is."
        ),
        "specific_headers": ["Atom", "Active series", "Tag", "Expanded tag", "Lift", "Active rate vs base"],
        "no_tag": "(no candidate tag)",
        "generic_h2": "The most generic atoms",
        "generic_p": (
            "The atoms with the most active series (up to {max_active} of {n_series}) all have "
            "a lift close to 1: they activate for almost everything, so no single tag "
            "distinguishes them. These are not \"unnamed\" for lack of trying, they are "
            "structural atoms (shared across most series) rather than concept atoms."
        ),
        "generic_headers": ["Atom", "Active series", "Closest tag", "Expanded tag", "Lift", "Active rate vs base"],
        "deepdive_h2": "Well-known macro tags",
        "deepdive_p1": (
            "The tables above rank atoms by their single best tag, found among the top 5 "
            "candidates saved per atom. A common tag like \"usa\" or \"unemployment\" can lose "
            "that spot to a rarer, more surprising tag even on an atom it is genuinely related "
            "to, so this section fixes ten well-known tags and checks all 128 atoms directly for "
            "each one, reporting both lift (how enriched the tag is inside the active group) and "
            "recall (what share of all series with that tag the atom actually covers)."
        ),
        "deepdive_p2": (
            "All ten tags are best covered, by recall, by the same three or four background "
            "atoms already listed above (92 to 99% recall, lift close to 1): they activate on "
            "almost every series, so they trivially cover almost every tag too, without encoding "
            "anything about it. The tables below exclude those background atoms and show the "
            "atoms that are actually specific to each tag."
        ),
        "deepdive_categories_h3": "The four verdict categories",
        "deepdive_categories": [
            ("Concentrated", "recall ≥ 50% and lift ≥ 10: a single atom dominates and is clearly specific. Only then do we call it \"the atom for this concept\"."),
            ("Broad, not concentrated", "recall ≥ 50% but lift < 10: the atom activates on almost every series with the tag, but also on many others that don't carry it. It is a wider shared concept, not this exact tag."),
            ("Fragmented", "no atom reaches 50% recall, but at least one has lift ≥ 5: the concept exists in the dictionary, spread across several atoms."),
            ("Absent", "not even that: no atom, outside the background ones, has lift ≥ 5 for this tag."),
        ],
        "series_word": "series",
        "of_corpus": "of the corpus",
        "cited_note": "The highlighted atom is the one cited in the verdict above.",
        "no_atoms": "No atom with useful recall or lift outside the background ones.",
        "deepdive_headers": ["Atom", "Active series", "Recall", "Lift"],
        "verdicts": {
            "concentrado": "concentrated: atom {atom} alone covers {recall:.0f}% of the series with this tag, with a lift of {lift:.1f}",
            "amplio": (
                "broad, not concentrated: atom {atom} covers {recall:.0f}% of the series with "
                "this tag, but its lift ({lift:.1f}) is low, so it also activates on many series "
                "that do not carry this tag: it is a wider shared concept, not this exact tag"
            ),
            "fragmentado": (
                "fragmented: the most informative atom ({atom}, lift {lift:.1f}) only covers "
                "{recall:.0f}% of the series with this tag, the rest is spread across other atoms"
            ),
            "ausente": "absent: no atom, outside the background ones, has a lift of 5 or higher for this tag",
        },
        "reading_h2": "Reading",
        "reading1": (
            "The dictionary splits into two clearly distinct groups: a small core of very rare, "
            "very specific atoms (active on tens or hundreds of series, lift in the hundreds or "
            "thousands: \"growth\", \"food stamps\", \"chile\", specific indices) and a handful "
            "of almost-always-active atoms (100k+ series, lift ~1) that do not correspond to any "
            "specific tag, but to structure shared across almost the whole base. Between those "
            "two extremes, {n_selective} of {dict_size} atoms ({pct_selective:.0f}%) have a tag "
            "with lift 5 or higher: the part of the dictionary that can confidently be called "
            "\"encodes this concept\", neither so rare that it doesn't generalise nor so generic "
            "that it says nothing."
        ),
        "reading2": (
            "With well-known macro tags, the pattern gets more nuanced and four groups show up, "
            "not two. GDP is the only clean case (genuinely concentrated): a single atom (75) "
            "covers 91% of its 152 series with a lift of 121, a rare and coherent concept that "
            "the SAE does learn as its own unit. A second group, broad but not concentrated "
            "(cpi, inflation, price index, trade, housing) has an atom with high recall (69 to "
            "95%) but a lift of only 2 to 3: it activates on most series with that tag, but also "
            "on very many others that don't carry it, so it is a wider concept (\"prices and "
            "trade\") rather than that exact tag. Unemployment and manufacturing are fragmented: "
            "there is an atom with high lift (17.3 and 5.9), but it covers little of the tag (33% "
            "and 2%), a sign that the concept exists in the dictionary but spread across several "
            "atoms, not that the SAE fails to distinguish it. And usa and employment are "
            "genuinely absent: no atom, background or not, has a useful lift for them, either too "
            "broad (employment) or too ubiquitous (usa, 88% of the corpus) for anything to "
            "specialise in them."
        ),
    }


def _bucket_counts(atoms: list[dict]) -> list[int]:
    buckets = [
        lambda r: r["n_active"] == 0,
        lambda r: r["n_active"] > 0 and not r["top_tags"],
        lambda r: r["top_tags"] and r["top_tags"][0]["lift"] < 5,
        lambda r: r["top_tags"] and 5 <= r["top_tags"][0]["lift"] < 20,
        lambda r: r["top_tags"] and 20 <= r["top_tags"][0]["lift"] < 100,
        lambda r: r["top_tags"] and r["top_tags"][0]["lift"] >= 100,
    ]
    return [sum(1 for r in atoms if pred(r)) for pred in buckets]


def _top_rows_html(atoms: list[dict], n: int, by: str, no_tag_label: str, tag_map: dict[str, str]) -> str:
    named = [r for r in atoms if r["top_tags"]]
    if by == "lift":
        ranked = sorted(named, key=lambda r: -r["top_tags"][0]["lift"])[:n]
    else:  # "n_active"
        ranked = sorted(atoms, key=lambda r: -r["n_active"])[:n]
    rows = []
    for r in ranked:
        t = r["top_tags"][0] if r["top_tags"] else None
        tag_cell = f"'{html.escape(t['tag'])}'" if t else no_tag_label
        expanded = html.escape(tag_map[t["tag"]]) if t and t["tag"] in tag_map else ""
        lift_cell = f"{t['lift']:.1f}" if t else "N/D"
        rate_cell = f"{t['active_rate']*100:.0f}% vs {t['base_rate']*100:.2f}%" if t else "N/D"
        rows.append(
            f'<tr><td class="n">{r["atom"]}</td><td class="n">{r["n_active"]}</td>'
            f"<td>{tag_cell}</td><td>{expanded}</td>"
            f'<td class="n">{lift_cell}</td><td class="n">{rate_cell}</td></tr>'
        )
    return "\n".join(rows)


def _tag_deep_dive_html(tag_deep_dive: dict, s: dict, tag_map: dict[str, str]) -> str:
    """Per-tag verdict, in 4 categories, plus a table that ALWAYS includes the atom cited in the
    verdict (highlighted), so no number in the text is left unverifiable in the table next to it.
    See `s["deepdive_categories"]` for the definition of the 4 categories.
    """
    items = []
    for tag, info in tag_deep_dive["tags"].items():
        tag_title = f"{tag} ({html.escape(tag_map[tag])})" if tag in tag_map else tag
        by_atom = {r["atom"]: r for r in info["top_atoms_by_lift"] + info["top_atoms_by_recall"]}
        specific = [r for r in by_atom.values() if r["n_active"] < _BACKGROUND_ATOM_THRESHOLD]
        by_lift_sorted = sorted(
            [r for r in info["top_atoms_by_lift"] if r["n_active"] < _BACKGROUND_ATOM_THRESHOLD],
            key=lambda r: -r["lift"],
        )

        best_recall = max(specific, key=lambda r: r["recall"]) if specific else None
        lift_qualifying = [r for r in specific if r["lift"] >= 5]

        cited: dict | None
        if best_recall and best_recall["recall"] >= 0.5 and best_recall["lift"] >= 10:
            cited = best_recall
            verdict = s["verdicts"]["concentrado"].format(atom=cited["atom"], recall=cited["recall"] * 100, lift=cited["lift"])
        elif best_recall and best_recall["recall"] >= 0.5:
            cited = best_recall
            verdict = s["verdicts"]["amplio"].format(atom=cited["atom"], recall=cited["recall"] * 100, lift=cited["lift"])
        elif lift_qualifying:
            cited = max(lift_qualifying, key=lambda r: r["recall"])
            verdict = s["verdicts"]["fragmentado"].format(atom=cited["atom"], recall=cited["recall"] * 100, lift=cited["lift"])
        else:
            cited = None
            verdict = s["verdicts"]["ausente"]

        table_atoms = list(by_lift_sorted[:3])
        if cited and cited["atom"] not in {r["atom"] for r in table_atoms}:
            table_atoms.append(cited)
        rows = "\n".join(
            (
                f'<tr{" class=\"hi\"" if cited and r["atom"] == cited["atom"] else ""}>'
                f'<td class="n">{r["atom"]}</td><td class="n">{r["n_active"]}</td>'
                f'<td class="n">{r["recall"]*100:.1f}%</td><td class="n">{r["lift"]:.1f}</td></tr>'
            )
            for r in table_atoms
        )
        note = (
            f'<p class="note">{s["cited_note"]}</p>'
            if cited and cited["atom"] not in {r["atom"] for r in by_lift_sorted[:3]}
            else ""
        )
        h = s["deepdive_headers"]
        items.append(f"""
  <h3>{tag_title}</h3>
  <p>{info['n_series_with_tag']} {s['series_word']} ({info['base_rate']*100:.1f}% {s['of_corpus']}). {verdict.capitalize()}.</p>
  <div class="scroll">
    <table>
      <thead><tr><th>{h[0]}</th><th>{h[1]}</th><th>{h[2]}</th><th>{h[3]}</th></tr></thead>
      <tbody>
{rows if rows else f'<tr><td colspan="4">{s["no_atoms"]}</td></tr>'}
      </tbody>
    </table>
  </div>
  {note}""")
    return "\n".join(items)


def render_lang(data: dict, tag_deep_dive: dict | None, lang: str, tag_map: dict[str, str]) -> str:
    s = strings(lang)
    atoms = data["atoms"]
    dict_size = data["dict_size"]
    n_dead = data["n_dead_atoms"]
    n_alive = dict_size - n_dead
    n_named = data["n_named_atoms"]
    n_selective = sum(1 for r in atoms if r["top_tags"] and r["top_tags"][0]["lift"] >= 5)
    n_series = data["n_series"]
    n_with_tags = data["n_with_tags"]

    defs_html = "\n".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in s["defs"])
    cats_html = "\n".join(
        f"<dt>{k}</dt><dd>{v}</dd>" for k, v in s["deepdive_categories"]
    )

    bh = s["bucket_headers"]
    bucket_rows = "\n".join(
        f'<tr><td>{label}</td><td class="n">{count}</td><td class="n">{count/dict_size*100:.1f}%</td></tr>'
        for label, count in zip(s["bucket_labels"], _bucket_counts(atoms))
    )
    sh = s["specific_headers"]
    specific_rows = _top_rows_html(atoms, 15, "lift", s["no_tag"], tag_map)
    gh = s["generic_headers"]
    generic_rows = _top_rows_html(atoms, 10, "n_active", s["no_tag"], tag_map)

    deep_dive_section = ""
    if tag_deep_dive:
        deep_dive_section = f"""
<section>
  <h2>{s['deepdive_h2']}</h2>
  <p>{s['deepdive_p1']}</p>
  <p>{s['deepdive_p2']}</p>
  <h3>{s['deepdive_categories_h3']}</h3>
  <dl class="defs">
{cats_html}
  </dl>
{_tag_deep_dive_html(tag_deep_dive, s, tag_map)}
</section>"""

    return f"""
<header>
  <div class="kicker">{s['kicker']}</div>
  <h1>{s['h1']}</h1>
  <p class="standfirst">{s['intro1'].format(dict_size=dict_size)}</p>
  <p class="standfirst">{s['intro2'].format(n_series=n_series, n_with_tags=n_with_tags, pct_tags=n_with_tags/n_series*100)}</p>
</header>

<section>
  <h2>{s['method_h2']}</h2>
  <p>{s['method_p1']}</p>
  <p>{s['method_p2']}</p>
  <h3>{s['defs_h3']}</h3>
  <dl class="defs">
{defs_html}
  </dl>
</section>

<section>
  <h2>{s['bucket_h2']}</h2>
  <p>{s['bucket_p'].format(n_dead=n_dead, dict_size=dict_size, n_alive=n_alive, n_named=n_named, n_selective=n_selective)}</p>
  <div class="scroll">
    <table>
      <thead><tr><th>{bh[0]}</th><th>{bh[1]}</th><th>{bh[2]}</th></tr></thead>
      <tbody>
{bucket_rows}
      </tbody>
    </table>
  </div>
</section>

<section>
  <h2>{s['specific_h2']}</h2>
  <p>{s['specific_p']}</p>
  <div class="scroll">
    <table>
      <thead><tr><th>{sh[0]}</th><th>{sh[1]}</th><th>{sh[2]}</th><th>{sh[3]}</th><th>{sh[4]}</th><th>{sh[5]}</th></tr></thead>
      <tbody>
{specific_rows}
      </tbody>
    </table>
  </div>
</section>

<section>
  <h2>{s['generic_h2']}</h2>
  <p>{s['generic_p'].format(max_active=max(r['n_active'] for r in atoms), n_series=n_series)}</p>
  <div class="scroll">
    <table>
      <thead><tr><th>{gh[0]}</th><th>{gh[1]}</th><th>{gh[2]}</th><th>{gh[3]}</th><th>{gh[4]}</th><th>{gh[5]}</th></tr></thead>
      <tbody>
{generic_rows}
      </tbody>
    </table>
  </div>
</section>
{deep_dive_section}
<section>
  <h2>{s['reading_h2']}</h2>
  <div class="key">
    <p>{s['reading1'].format(n_selective=n_selective, dict_size=dict_size, pct_selective=n_selective/dict_size*100)}</p>
  </div>
  <div class="key">
    <p>{s['reading2']}</p>
  </div>
</section>"""


def build_html(data: dict, tag_deep_dive: dict | None, tag_map: dict[str, str]) -> str:
    es = render_lang(data, tag_deep_dive, "es", tag_map)
    en = render_lang(data, tag_deep_dive, "en", tag_map)

    return f"""<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1"><style>:root{{color-scheme:light}}body{{margin:0;padding:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif;background:#faf9f5;color:#141413}}img{{max-width:100%}}[hidden]:not([hidden=until-follow i]){{display:none!important}}</style></head><body data-lang="es">
<title>Concept naming: SAE dictionary</title>
<style>{CSS}</style>
<div class="wrap">
  <div class="langbar">
    <div class="langtoggle">
      <button type="button" data-setlang="es" class="active">ES</button>
      <button type="button" data-setlang="en">EN</button>
    </div>
  </div>
  <div class="langblock" data-lang="es">{es}</div>
  <div class="langblock" data-lang="en">{en}</div>
</div>
<script>
(function () {{
  var KEY = "concept_report_lang";
  var saved = "es";
  try {{ saved = localStorage.getItem(KEY) || "es"; }} catch (e) {{}}
  document.body.setAttribute("data-lang", saved);
  var buttons = document.querySelectorAll(".langtoggle button");
  buttons.forEach(function (btn) {{
    btn.classList.toggle("active", btn.dataset.setlang === saved);
    btn.addEventListener("click", function () {{
      document.body.setAttribute("data-lang", btn.dataset.setlang);
      buttons.forEach(function (b) {{ b.classList.toggle("active", b === btn); }});
      try {{ localStorage.setItem(KEY, btn.dataset.setlang); }} catch (e) {{}}
    }});
  }});
}})();
</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", type=Path,
        default=Path("results/concepts/concept_naming_neurosym-cbf-kaggle_ctx30.json"),
    )
    parser.add_argument(
        "--tag-deep-dive", type=Path,
        default=Path("results/concepts/tag_deep_dive_neurosym-cbf-kaggle_ctx30.json"),
        help="Output of scripts/17_concept_tag_deep_dive.py; skipped if it does not exist",
    )
    data_config = load_yaml_config("configs/data.yaml")
    parser.add_argument(
        "--notes-lookup", type=Path,
        default=data_dir() / data_config["paths"]["tags_dir"] / "tags_notes_lookup.json",
        help="Short tag -> expanded text lookup (download_tags.py); skipped if it does not exist",
    )
    parser.add_argument("--output", type=Path, default=None, help="Defaults to next to the input")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    tag_deep_dive = json.loads(args.tag_deep_dive.read_text()) if args.tag_deep_dive.exists() else None
    if tag_deep_dive is None:
        logger.warning("%s not found, the well-known macro tags section is skipped", args.tag_deep_dive)
    tag_map = json.loads(args.notes_lookup.read_text()) if args.notes_lookup.exists() else {}
    if not tag_map:
        logger.warning("%s not found, the expanded-tag column will be left empty", args.notes_lookup)
    html_out = build_html(data, tag_deep_dive, tag_map)

    output = args.output or args.input.parent / "report_concept_naming.html"
    output.write_text(html_out, encoding="utf-8")
    logger.info("Report -> %s", output)


if __name__ == "__main__":
    main()
