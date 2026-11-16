import pytest
from docx import Document

from clean_transcripts import clean_data, normalize_folder, normalize_text, remove_interviewer, remove_text


def write_docx(path, paragraphs):
    doc = Document()
    for text in paragraphs:
        doc.add_paragraph(text)
    doc.save(path)


def test_clean_data_keeps_interviewee_turns(tmp_path):
    src, out = tmp_path / "original", tmp_path / "text"
    src.mkdir()
    write_docx(
        src / "05_speaker.docx",
        [
            "A: era muy bonito (risas) el barrio",
            "B: ¿y ahora?",
            "A: ahora está • bien",
            "B: fin",
        ],
    )

    clean_data(str(src), str(out))

    assert (out / "05_speaker.txt").read_text(encoding="utf-8") == "era muy bonito el barrio\nahora está  bien"


def test_clean_data_keeps_preposition_a(tmp_path):
    src, out = tmp_path / "original", tmp_path / "text"
    src.mkdir()
    write_docx(src / "05_speaker.docx", ["A: A mí me gusta. A veces sí", "B: fin"])

    clean_data(str(src), str(out))

    assert (out / "05_speaker.txt").read_text(encoding="utf-8") == "A mí me gusta. A veces sí"


def test_clean_data_reads_b_turns_when_b_speaks_more(tmp_path):
    src, out = tmp_path / "original", tmp_path / "text"
    src.mkdir()
    write_docx(src / "TrCTG_H21.docx", ["A: ¿y usted?", "B: yo vivo aquí hace años", "A: fin"])

    clean_data(str(src), str(out))

    assert (out / "TrCTG_H21.txt").read_text(encoding="utf-8") == "yo vivo aquí hace años"


def test_clean_data_keeps_informant_section_of_compilation(tmp_path):
    src, out = tmp_path / "original", tmp_path / "text"
    src.mkdir()
    write_docx(
        src / "34-Informante.docx",
        [
            "Hombre de 34 años",
            "Informante Nº 19",
            "A: soy José y hablo mucho aquí",
            "B: fin",
            "Mujer de 43 años",
            "Informante Nº 34",
            "A: soy yo y hablo mucho aquí",
            "B: fin",
        ],
    )

    clean_data(str(src), str(out))

    assert (out / "34-Informante.txt").read_text(encoding="utf-8") == "soy yo y hablo mucho aquí"


preseea_header = [
    "MEDE_H01_1",
    '<Hablante id="hab1" codigo_hab="I" papel="informante"/>',
    '<Hablante id="hab2" codigo_hab="E" papel="entrevistador"/>',
    "</Hablantes> </Trans>",
    "PRESEEA - MEDELLÍN",
    "E: Entrevistador",
    "I: Informante",
]


def test_clean_data_preseea_informant_turns(tmp_path):
    src, out = tmp_path / "original", tmp_path / "text"
    src.mkdir()
    write_docx(
        src / "MEDE_H01_1.docx",
        preseea_header
        + [
            "E:\t¿le parece que ha estado raro el clima?",
            "I.:\tsí / muy<alargamiento/> raro // <énfasis>mucho</énfasis> calor",
            "y lluvia E: ¿sí? I: sí señor",
            "E. ¿y el barrio?",
            "bonito <risas/> (risas)",
            "E: bueno",
            "otra cosa",
        ],
    )

    clean_data(str(src), str(out))

    assert (out / "MEDE_H01_1.txt").read_text(encoding="utf-8").split("\n") == [
        "sí, muy raro. mucho calor y lluvia",
        "sí señor",
        "bonito",
    ]


def test_clean_data_reads_dialogue_tables(tmp_path):
    src, out = tmp_path / "original", tmp_path / "text"
    src.mkdir()
    doc = Document()
    for text in preseea_header:
        doc.add_paragraph(text)
    row = doc.add_table(rows=1, cols=2).rows[0]
    row.cells[0].text = "1\n2"
    row.cells[1].text = "E.: ¿cómo está?\nI.: bien gracias"
    doc.save(src / "G2I1M20.docx")

    clean_data(str(src), str(out))

    assert (out / "G2I1M20.txt").read_text(encoding="utf-8") == "bien gracias"


def test_remove_text_only_listed_files(tmp_path):
    (tmp_path / "a.txt").write_text("uno dos tres", encoding="utf-8")
    (tmp_path / "b.txt").write_text("uno dos tres", encoding="utf-8")

    remove_text(str(tmp_path), 2, ["a.txt"])

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "tres"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "uno dos tres"


def test_remove_text_drops_leading_words(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("uno dos tres\ncuatro cinco\nseis\n", encoding="utf-8")

    remove_text(str(tmp_path), 4)

    assert f.read_text(encoding="utf-8") == "cinco\nseis\n"


def test_strip_speaker_labels():
    from clean_transcripts import strip_speaker_labels

    assert (
        strip_speaker_labels("de mayagüez J: -e bien yo LMP: - si E: ¿cómo? A las")
        == "de mayagüez e bien yo si ¿cómo? A las"
    )


def test_clean_turns_drops_interviewer_line():
    from clean_transcripts import clean_turns

    text = (
        "".join(f"línea {i}\n" for i in range(10))
        + "E: ¿y el barrio? ¿cómo era?\nera muy tranquilo\nB: habrá títeres\nG.: hasta luego\n"
    )
    assert clean_turns(text).split("\n")[10:] == ["", "era muy tranquilo", "", "", ""]


def test_clean_turns_inline_keeps_answer_after_question():
    from clean_transcripts import clean_turns

    assert clean_turns("comento? E: ¿cómo será Tunja? difícil, difícil") == "comento?  difícil, difícil"


def test_clean_turns_keeps_participant_label_turns():
    from clean_transcripts import clean_turns

    text = "inicio " + "J: -sí " * 20 + "X: -(tercera voz) susan J: -bueno"
    assert clean_turns(text).split() == ("inicio " + "sí " * 20 + "bueno").split()
    assert clean_turns(
        "sabía. E: ¿Y ya sabía inglés? I: Mi inglés no era muy bueno. E: Ah-ha. I: El clima"
    ).split() == ["sabía.", "Mi", "inglés", "no", "era", "muy", "bueno.", "El", "clima"]


def test_clean_turns_drops_third_voice_under_participant_label():
    from clean_transcripts import clean_turns

    text = "inicio " + "F: -sí " * 20 + "F: -(tercera voz) sigue F: -yo tengo hermano"
    assert clean_turns(text).split()[-4:] == ["sí", "yo", "tengo", "hermano"]


def test_clean_turns_inline_interviewer_turn_ends_after_questions():
    from clean_transcripts import clean_turns

    text = "los niños E.: a, ¡ve que si!, se nota ¿no? ¿cuántos años llevan juntos?, me decía hace veinte años"
    assert clean_turns(text).split() == ["los", "niños", ",", "me", "decía", "hace", "veinte", "años"]


def test_clean_turns_line_based_drops_frequent_other_speaker():
    from clean_transcripts import clean_turns

    text = "".join(f"él habla {i}\nA.: ella responde {i}\n" for i in range(25))
    assert "ella" not in clean_turns(text)
    assert clean_turns(text).count("él habla") == 25


pad = "".join(f"relleno {i}\n" for i in range(10))


@pytest.mark.parametrize(
    ("lines", "kept"),
    [
        (["E. ¿en qué barrio vive?", "en Antonio Nariño"], ["", "en Antonio Nariño"]),
        (["E: ¿y la familia?", "bien"], ["", "bien"]),
        (["e: ¿y cómo es?", "bonito"], ["", "bonito"]),
        (["E ¿y cómo es?", "bonito"], ["", "bonito"]),
        (["I.. ¿el peor susto de mi vida? un día"], ["¿el peor susto de mi vida? un día"]),
        (["B. sí señor"], ["sí señor"]),
        # lowercase "e" before a question or period is the filler "eh"
        (["e ¿cómo se dice?", "eso"], ["e ¿cómo se dice?", "eso"]),
        # unlabeled questions stay, even ones that look put to the participant
        (["¿y por qué le gustaría ir a vivir allá?", "sí, mucho también"], None),
        (["¿pero cercano?", "¿cercano?, sí una vez"], None),
        (["¿cuentos de historias?", "Sí, también."], None),
    ],
)
def test_interviewer_line_based(lines, kept):
    assert remove_interviewer(pad + "\n".join(lines)).split("\n")[10:] == (kept or lines)


@pytest.mark.parametrize(
    "text",
    [
        "amiga? bueno ahora vamos a la parte de tu profesión ¿a que te dedicas? ¿en que te gustaría trabajar? en un banco",
        "¿tienes amigos? ¿como eran tus amigos de infancia? pues muchos",
        "me dice el profe ¿y usted cultiva papa? le dije claro",
    ],
)
def test_interviewer_inline_unlabeled_kept(text):
    assert remove_interviewer(text) == text


@pytest.mark.parametrize(("args", "kept"), [([], 1180), (["--start-minute", "5"], 1790), (["--skip-words", "0"], 2300)])
def test_cli_start_minute(tmp_path, args, kept):
    import subprocess
    import sys
    from pathlib import Path

    src, out = tmp_path / "original", tmp_path / "text"
    src.mkdir()
    write_docx(src / "05_speaker.docx", ["A: " + "palabra " * 2300, "B: fin"])
    script = Path(__file__).parent.parent / "pipeline" / "clean_transcripts.py"

    subprocess.run([sys.executable, script, "--input", src, "--output", out, *args], check=True, capture_output=True)

    assert len((out / "05_speaker.txt").read_text(encoding="utf-8").split()) == kept


@pytest.mark.parametrize(
    ("text", "clean"),
    [
        ("818 PRESSEA - CARTAGENA NIVEL 2 - EDAD 3\nEsos son, ya", "Esos son, ya"),
        ("y era muy bonito CALI_H12_3-45m", "y era muy bonito"),
        ("eso era [update 2020] muy bueno", "eso era muy bueno"),
        ("eso [T] era muy bueno", "eso era muy bueno"),
        ("eso [[T] algo] era muy bueno", "eso era muy bueno"),
        ("profetas,**OJO this example goes in the manuscrito y ya", "profetas, y ya"),
        ("y J -pues sí", "y pues sí"),
        ("era muy—bonito", "era muy bonito"),
        ("era muy -- bonito", "era muy bonito"),
        ("sí!! claro._ ya", "sí! claro. ya"),
    ],
)
def test_normalize_drops_notes_and_marks(text, clean):
    assert normalize_text(text) == clean


@pytest.mark.parametrize(
    "text",
    [
        "era muy ruido =“eructo bonito",
        'era muy risas = "jaja" bonito',
        "era muy xxx bonito",
        "era muy [xxx] bonito",
        "era muy jajaja bonito",
    ],
)
def test_normalize_noise_becomes_pause(text):
    # a pause keeps "muy" from looking adjacent to the host past the noise
    assert normalize_text(text) == "era muy, bonito"


def test_normalize_line_numbers():
    text = "\n".join(f"{n}) dijo 300 cosas" for n in range(100, 125)) + "\n125 suelto\n999 años"

    lines = normalize_text(text).split("\n")

    assert lines[0] == "dijo 300 cosas"
    assert lines[-2:] == ["suelto", "999 años"]


@pytest.mark.parametrize(
    ("text", "clean"),
    [
        ("en la iglesia \"La Unción' Sí", 'en la iglesia "La Unción" Sí'),
        ("vamos pa' allá", "vamos pa' allá"),
        ("dijo ' no", "dijo no"),
        ("s´ señor", "sí señor"),
    ],
)
def test_normalize_quotes(text, clean):
    assert normalize_text(text) == clean


@pytest.mark.parametrize(
    ("text", "clean"),
    [
        ("YO no QUISIERA que USTED se FUERA", "Yo no quisiera que usted se fuera"),
        ("VOY A EXPLICAR", "Voy a explicar"),
        ("eso. MUY bonito", "eso. Muy bonito"),
        ("fui al SENA y al MIO", "fui al SENA y al MIO"),
        ("el siglo XX y el CAi", "el siglo XX y el CAI"),
    ],
)
def test_normalize_lowercases_caps_keeps_acronyms(text, clean):
    assert normalize_text(text) == clean


@pytest.mark.parametrize(
    ("text", "clean"),
    [
        ("la vece s que fui entonce s", "la veces que fui entonces"),
        ("Entonce s fui", "Entonces fui"),
        ("la s. ma la va ya", "la s. ma la va ya"),
        ("es fácil y tenia tambien comodo", "es fácil y tenía también cómodo"),
        ("Tambien fui", "También fui"),
        ("yo practico y limite", "yo practico y limite"),
    ],
)
def test_normalize_splits_and_accents(text, clean):
    assert normalize_text(text) == clean


def test_normalize_folder_returns_changed(tmp_path):
    (tmp_path / "a.txt").write_text("MUY bonito", encoding="utf-8")
    (tmp_path / "b.txt").write_text("muy bonito", encoding="utf-8")

    assert normalize_folder(tmp_path) == ["a.txt"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "Muy bonito"
