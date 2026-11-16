# Transcript cleaning

`pipeline/clean_transcripts.py` turns the docx transcripts into `data/text` (one turn per line), the text the model trains and predicts on. Counts are firings on the 310-transcript corpus when each rule was added; tests are in `tests/test_clean_transcripts.py`.

Only text marked by a speaker label is removed. Unlabeled questions stay, even when they look addressed to the participant, since without audio they can't be told apart from the participant's own rhetorical questions.

| Case | Handling |
| --- | --- |
| docx text | paragraphs plus table cells (Tunja generations 2-3 put the dialogue in a table); line-number cells skipped |
| compilation docx (several *Informante Nº* sections, e.g. `34-Informante`) | only the section whose number is the first number in the file name |
| Cartagena turns | `A:` or `B:` turns, whichever has more words (the informant; `B` only in `TrCTG_H21`), other labeled turns dropped |
| PRESEEA header (Medellín, Cali, Tunja) | everything through the last `codigo_hab` line dropped, plus the `E: Entrevistador` legend |
| PRESEEA turns | `I` turns kept; labels `E:`, `I.:`, `e:`, colon-less line-initial `E.`/`I.` (Tunja), mid-line `E:`; every other header code (`A1`, `B`, ...) dropped |
| PRESEEA unlabeled line after an interviewer question (`E: ¿...?` then a line with no label) | informant's answer with its label missing (~360 words) |
| PRESEEA markup (`<alargamiento/>`, `<énfasis>...</énfasis>`, `<tiempo .../>`), stray `<`, `>`, `====` | tags removed, enclosed words kept |
| PRESEEA pauses `/`, `//` | `,` and `.` |
| parentheticals (*(risas)*), `•`, `·` | removed |
| one-line transcripts (LAPR, TUNJ) with turn labels | participant label = `I` if present, else the label used 20+ times; other labeled turns (interviewer, `X` third voice) dropped up to the next label, or through their question(s) when the participant answers on without a label (`E.: ... ¿cuántos años llevan juntos?, me decía ...`) |
| line-based transcripts (TCali, MEDE) | every labeled line is someone else (interviewer, relative, radio) and is dropped |
| *(tercera voz)* after the participant's label | dropped as a third voice |
| interviewer line labels `E.`, `E..`, `E:`, `e:`, `E ¿` | whole line dropped; lowercase `e ¿`, `e.` are the filler *eh* and stay |
| participant line labels `I.`, `i:`, `B.` | label dropped, words kept |
| remaining participant labels (`J: -pues`) | label dropped |
| `MEDE_H21_3` docx holds all of `MEDE_H21_2` pasted inside it (8,734 words, its own header says 33:47 and 5,186 words) | cut by hand from `data/text` on 2026-11-02 (original in `data/duplicates/MEDE_H21_3_pre_slice/`). The docx is unchanged, so re-cleaning it brings the copy back: cut it again |
| first 1,120 informant words (~ minute 10; `--start-minute 5` drops 510 instead) | dropped (`--start-minute`, or an exact `--skip-words`), only in files cleaned this run. Minute 10 is where the PRESEEA subject-pronoun (10-20) and attenuation (10-40) guides start, since the opening is the most formal stretch (see [Interview start](#interview-start)); 1,120 = median informant words at the 10:00 stamp in the 190 time-stamped MEDE/TCali files (middle half 952-1,269). One word count for every file because Tunja and Cartagena have no stamps. LAPR (no docx, already cut at 350) cut 770 more by hand on 2026-11-01. Interviews shorter than the cut are left empty, no exemption (`MEDE_H01_1`, 23 min) |

## Text normalization (`normalize_text`, 2026-11-15)

Runs after the cut (so the cut is unchanged), on files cleaned that run, or on all of `data/text` with `--normalize-only`. Goal: the model ([finetune.md](finetune.md)) trains and predicts on clean text. Applied 2026-11-15: 180 of 310 files changed, 2,738 word-level edits. Counts below are firings on that `data/text`.

| Case | Handling | Count |
| --- | --- | --- |
| Cartagena page headers (`818 PRESSEA - CARTAGENA NIVEL 2 - EDAD 3`) | line dropped | 20 |
| Cali file-name tails (`CALI_H12_3-45m`), `[update ...]`, `[document]`, `* *` | dropped | 4, 9, 2 |
| annotator notes (`**OJO this example goes in the manuscrito`, 3), `[T]`/`[[T] ...]` edit marks (63), `{...)`, `J -`, `h!!` | dropped | 3, 63, 8 |
| noises (`ruido = "eructo`), laughter (`risas = "..."`, `entre_risas`, *jajaja*, *jejeje*), unintelligible `xxx`/`[xxx]` | a comma, so no intensifier looks adjacent to a host across them (*muy xxx bonito* -> *muy, bonito*) | 4, 202, 150 |
| printed line numbers (`LAPR_M07`: `143)`, bare `142` next to a marked one) | dropped; only in a file with 20+ `NNN)` | 70 |
| leftover brackets `[ ] ( ) { }` (mostly lone `)`) | dropped, words kept | 1,096 |
| `--`, em/en dashes (*muy—bonito*) | space | 72 |
| closing `'`/`''` of a `"` quote (*"La Unción'*) | `"`; elisions *pa'*, *pal'*, *ta'*, *na'*, *to'*, *po'*, *pue'* kept; lone `'` dropped; `s´` -> *sí*; `!!` -> `!`, `._` -> `.` | 52 |
| words in capitals for emphasis or verb marking (*YO no QUISIERA que USTED*, *MUY bonito*) | lowercased, sentence-initial capital kept; 105 acronyms (*SENA*, *MIO*, *ICFES*, *CAI*...), roman numerals and their typos (*CAi* -> *CAI*, *EPEses*) kept | 1,604 |
| words split by a space (*entonce s*, *vece s*, *cuida do*; 117 pairs, mostly MEDE) | joined; restarts that are real word pairs (*ma la*, *va ya*) left alone | 151 |
| missing accents on whole words that are never another word in the corpus (*tenia*, *tambien*, *comodo*, *despues*; 139 forms) | accent restored, case kept; verb forms like *practico*, *limite*, *publica* left alone | 341 |

Originals in `data/text_pre_normalize/`.

## Corpus files

- Four numbered Cartagena files (`19-Jose`, `20-Evelyn`, `29-Rodolfo`, `30-Edelfa`) are the same speakers as `TrCTG_*` files. The `TrCTG_*` copies are kept; the numbered ones sit in `data/duplicates/` and must not go back into `data/original` or `data/text`.
- LAPR (15 files) has no docx; its `data/text` files are kept as they are. Re-clean them once the docx are back.
- Interviews shorter than the minute-10 cut come out empty (`MEDE_H01_1`) and give no rows.
- Only speaker-label-marked text is removed as interviewer or other turns; unlabeled interviewer speech stays.

## Interview start

The 1,120-word cut is minute 10 of the interview, converted to words from this corpus's own time stamps. Minute 10 is where the PRESEEA guides for subject pronouns and attenuation start; the others start at minute 5 or 0. The first 10 minutes are treated as the most formal stretch (Guerrero González et al. 2024), the observer's paradox of Labov (1972).

- Labov, W. (1972). *Sociolinguistic Patterns*. Philadelphia: University of Pennsylvania Press, p. 209 (observer's paradox).
- Guerrero González, S. & Manjón-Cabeza Cruz, A. (2022). *Guía PRESEEA de estudio de la expresión del sujeto pronominal*. Minutes 10-20. <https://preseea.uah.es/sites/default/files/2022-05/Gu%C3%ADa%20PRESEEA%20de%20estudio%20de%20la%20expresi%C3%B3n%20del%20sujeto%20pronominal_Guerrero%20y%20Manj%C3%B3n%20(2022).pdf>
- Cestero Mancera, A. M. & Rodríguez Alfano, L. (2021). *Guía PRESEEA de estudio de la atenuación*. Minutes 10-40. <https://preseea.uah.es/sites/default/files/2022-02/Gu%C3%ADa%20PRESEEA%20de%20estudio%20de%20la%20atenuaci%C3%B3n_Cestero%20y%20Rodr%C3%ADguez%20%282021%29.pdf>
- Cestero Mancera, A. M. (2022). *Guía PRESEEA de estudio de apéndices interrogativos de control de contacto*. Minutes 5-40: 5-15 formal, 15-30 middle, 30-40 informal register. <https://preseea.uah.es/sites/default/files/2022-02/Gu%C3%ADa%20PRESEEA%20de%20estudio%20de%20ap%C3%A9ndices%20interrogativos%20de%20control%20de%20contacto_Cestero%20(2022).pdf>
- Samper Padilla, J. A., Malaver Arguinzones, I. & Samper Hernández, M. (2021). *Guía PRESEEA de estudio de la /d/ intervocálica*. First, middle and last five minutes. <https://preseea.uah.es/sites/default/files/2022-02/Gu%C3%ADa%20PRESEEA%20de%20estudio%20de%20la%20d%20intervoc%C3%A1lica_Samper,%20Malaver%20y%20Samper%20(2021).pdf>
- Guerrero González, S., González Riffo, J., Espinoza Díaz, V., Ibarra Herrera, D. & Gajardo Moller, C. (2024). Evidencialidad de acervo común: patrones sociopragmáticos en el habla de Santiago de Chile. *Philologica Canariensia* 30, 275-296. Minutes 5-35; the first 10 minutes are assumed closest to formal register. <https://ojsspdc.ulpgc.es/ojs/index.php/PhilCan/article/download/1662/1819?inline=1>

Word equivalent: informant words before the interpolated `<tiempo>` stamp, 190 MEDE and TCali files: 10:00 median 1,122 (middle half 952-1,269), used as 1,120; 5:00 median 512 (middle half 437-594), used as 510 by `--start-minute 5` (start of the interrogative-tag and evidentiality windows); 15:00 = 1,724. LAPR has no docx: for minute 5, cut 160 words from `data/text_pre_reclean/LAPR_*` (already cut at 350), not from `data/text`.
