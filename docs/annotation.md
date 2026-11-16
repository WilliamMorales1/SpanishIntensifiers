# Annotation guide

What a row is and how each column is filled, for hand-correcting model output before it goes into the training data ([finetune.md](finetune.md#the-correction-loop)). The model learned these conventions from the rule pipeline it replaced, so corrected rows should follow them too, or say in the dataset notes where they depart.

## Which tokens are rows

One row per adjective or adverb token that can take an intensifier, intensified or not. Not rows:

- comparatives and superlatives: *más grande*, *menos favorecidas*, *mucho más*, *lo más importante*, *el momento más feliz*; *mejor*, *peor*; *mayor*, *menor* unless intensified (*muy mayor* = elderly);
- *tan* X *como* (equality), *tan* X *que* result clauses (Bulgin et al. 2008:103 drop *so X that*), *qué/cuán tan* X (= how), *tan solo/pronto/siquiera*;
- fixed expressions: *bien hecho*, *mucho más/menos/mejor/peor/mayor/menor*;
- *tan*, *más*, *menos*, *muy* as hosts themselves;
- disfluency fragments (*muy bue muy bueno*: only the second is a row);
- words that are not really adjectives or adverbs here: discourse *bueno*, *claro*, *listo*, *cierto*, tag questions *¿cierto?*, function words (*tanto*, *hasta*, *como*, *digamos*).

Kept: *por/a lo menos* X (host bare), *más o menos* X (compromiser), negated hosts (marked in `Negation`).

## Columns

| column | fill with |
| --- | --- |
| `Adj` / `Adv` | the host as written, for *-ísimo* and prefixed forms the base (*riquísimo* -> *rico*, *archiconocido* -> *conocido*); the other one `/` |
| `Lex_Adj` | masculine singular lemma of an adjective host (*bonitas* -> *bonito*, *acogedora* -> *acogedor*, invariant *egoísta*, *feliz*) |
| `Int` | intensifier word next to the host (*muy*, *bien*, *tan*), or the bound one (`-ísimo`, `súper`, `re`, `archi`), or `/` |
| `Double` | a second intensifier before it (*muy muy*, *bien bien*), or `/` |
| `Int_type` | filled from `Int` by `pipeline/int_type.py`, table below; not learned, so corrections here are ignored |
| `Adj_type` | `attributive` (modifies a noun: *una casa muy bonita*), `predicative` (copula, *quedar*, *ponerse*, secondary predicates: *es muy bonita*, *se puso feo*, *lo veo bonito*, *veía eso todo raro*), `ambiguous` (no head in the segment: *muy normal .*, fillers); `/` on adverb rows |
| `Noun` | noun the adjective modifies (attributive) or its subject/object (predicative), only if it is in `Phr`, else `/` |
| `Negation` | *no*, *nunca*, *ni*, *tampoco*, *jamás* in a preverbal slot scoping over the host (*no es muy bueno*, *ni tan pobres*); not discourse *no* (*no , muy buena*) |
| `Verb`, `Inf_Verb` | verb, copula or auxiliary the host hangs on, as written and as its infinitive, only if in `Phr`; adverb rows too |

A *mucho* after a lexical verb belongs to the verb, not the host (*compartimos mucho juntos*: bare host). Dialectal *mucho* = *muy* (*era mucho chusco*) is an intensifier.

## Int_type

Quirk et al. (1985:589-590) as operationalized by Stratton & Sundquist (2022:388-389) and Stratton (2020). Emphasizers are not scalar and "did not count as an instance of intensification" (Stratton & Sundquist 2022:395): their rows keep `Int` but count as unintensified. The mapping is `pipeline/int_type.json`, applied after the model by `predict.py`; an `Int` missing from it gets `?`. Another classification is another JSON file: `python pipeline/int_type.py <table> --map mine.json` refills an existing table, no retraining.

| Int_type | Int |
| --- | --- |
| maximizer | *absolutamente, completamente, totalmente, plenamente, extremadamente, sumamente* |
| booster | *muy, bien, mucho, harto, bastante, demasiado, súper, re, requete(-), recontra(-), archi-, -ísimo, -érrimo, realmente, verdaderamente, increíblemente, tan*, exclamative *más* |
| approximator | *casi* |
| compromiser | *más o menos* |
| emphasizer | *especialmente, estrictamente* |

Judgement calls (see the README): *bastante*, *demasiado* boosters (Colombian 'very'); *extremadamente*, *sumamente* maximizers (Quirk lists *extremely*); *realmente* booster (Tagliamonte). Negated rows are kept and marked, not dropped as in Stratton & Sundquist (2022:395).

## References

- Brown, E. L. & Cortés-Torres, M. (2013). Puerto Rican intensifiers: *Bien/muy* variables. In A. M. Carvalho & S. Beaudrie (eds.), *Selected Proceedings of the 6th Workshop on Spanish Sociolinguistics*, 11-19. Somerville, MA: Cascadilla Proceedings Project.
- Bulgin, J., Elford, N., Harding, L., Henley, B., Power, S. & Walters, C. (2008). So very really variable: Social patterning of intensifier use by Newfoundlanders online. *Linguistica Atlantica* 29, 101-115.
- D'Arcy, A. (2015). Stability, stasis and change. *Diachronica* 32, 449-493.
- Ito, R. & Tagliamonte, S. A. (2003). *Well* weird, *right* dodgy, *very* strange, *really* cool: Layering and recycling in English intensifiers. *Language in Society* 32(2), 257-279.
- Orozco, R., Morales, W. & Valenzuela, L. (2026). A variationist study of Spanish intensifiers. *Proceedings of the Linguistic Society of America*.
- Quirk, R., Greenbaum, S., Leech, G. & Svartvik, J. (1985). *A Comprehensive Grammar of the English Language*. London: Longman, pp. 589-590.
- Stratton, J. M. (2020). Adjective intensifiers in German. *Journal of Germanic Linguistics* 32(2), 183-215.
- Stratton, J. M. & Sundquist, J. D. (2022). A variationist sociolinguistic analysis of intensifiers in Oslo Norwegian. *Journal of Germanic Linguistics* 34(4), 385-419.
