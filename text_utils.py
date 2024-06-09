
_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i


class TextCleaner:
    def __init__(self):
        self.word_index_dictionary = dicts
        print(len(dicts))

    def __call__(self, text):
        indexes = []
        text = text.replace("(", "“")
        text = text.replace(")", "”")
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                save_to_txt("ouput.txt", char)
        return indexes
    
    def normalizer(self, text):
        text = _normalize_percentages(text)
        text = _normalize_time(text)
        text = _normalize_money(text)
        text = _normalize_am_pm_times(text)
        text = _normalize_numbers_with_letters(text)
        text = _normalize_numbers(text)
        text = _normalize_abbreviations(text)
        text = replace_punctuation(text)
        text = remove_aux_symbols(text)
        text = remove_punctuation_at_begin(text)
        text = collapse_whitespace(text)
        text = re.sub(r"([^\.,!\?\-…])$", r"\1.", text)
        return text


def save_to_txt(file_path, content):
    try:
        # Ler o conteúdo existente do arquivo
        with open(file_path, 'r', encoding='utf-8') as file:
            existing_content = file.readlines()
        
        # Verificar se o conteúdo já está presente
        if f"{content}\n" in existing_content:
            pass
        else:
            # Adicionar o novo conteúdo ao arquivo
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(f"{content}\n")
            print(f"Conteúdo '{content}' adicionado com sucesso.")
    except FileNotFoundError:
        # Se o arquivo não existir, criar e adicionar o conteúdo
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(f"{content}\n")
        print(f"Arquivo criado e conteúdo '{content}' adicionado com sucesso.")

def process_train_list(file_path, text_cleaner):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    second_column_texts = [line.split('|')[1].strip() for line in lines]

    for text in second_column_texts:
        text_cleaner(text)


import re
from num2words import num2words

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": ".",
    "…": ".",
    "$": ".",
    "“": "'",
    "”": "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
    "& ": " e ",
}

# Lista de pares (expressão regular, substituição) para abreviações em português do Brasil:
abbreviations = [
    (re.compile(r"\b%s\b" % re.escape(x[0]), re.IGNORECASE), x[1])
    for x in [
        ("sr", "senhor"),
        ("sra", "senhora"),
        ("dr", "doutor"),
        ("dra", "doutora"),
        ("prof", "professor"),
        ("eng", "engenheiro"),
        ("ltda", "limitada"),
        ("adv", "advogado"),
        ("etc.", "etcetera"),
        ("kb", "kilobyte"),
        ("gb", "gigabyte"),
        ("mb", "megabyte"),
        ("kw", "quilowatt"),
        ("mw", "megawatt"),
        ("gw", "gigawatt"),
        ("kg", "quilograma"),
        ("hz", "hertz"),
        ("khz", "quilo-hertz"),
        ("mhz", "mega-hertz"),
        ("ghz", "giga-hertz"),
        ("km", "quilômetro"),
        ("ltda", "limitada"),
        ("jan", "janeiro"),
        ("fev", "fevereiro"),
        ("mar", "março"),
        ("abr", "abril"),
        ("mai", "maio"),
        ("jun", "junho"),
        ("jul", "julho"),
        ("ago", "agosto"),
        ("set", "setembro"),
        ("out", "outubro"),
        ("nov", "novembro"),
        ("dez", "dezembro"),
        ("pág", "página"),
        ("págs", "páginas"),
        ("s.a", "sociedade anônima"),
        ("cia", "companhia"),
        ("etc", "et cetera"),
    ]
]


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    return replaced_text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()


def remove_punctuation_at_begin(text):
    return re.sub(r"^[,.!?]+", "", text)


def remove_aux_symbols(text):
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»\']+", "", text)
    return text


def _normalize_percentages(text):
    return re.sub(
        r"(\d+)%", lambda m: num2words(m.group(1), lang="pt") + " por cento", text
    )


def _normalize_time(text):
    def time_to_words(match):
        hours = int(match.group(1))
        minutes = int(match.group(2))
        hours_text = num2words(hours, lang="pt", to="cardinal")
        if minutes == 0:
            return f"{hours_text} hora" + ("s" if hours > 1 else "")
        minutes_text = num2words(minutes, lang="pt", to="cardinal")
        return (
            f"{hours_text} hora"
            + ("s" if hours > 1 else "")
            + f" e {minutes_text} minuto"
            + ("s" if minutes > 1 else "")
        )

    return re.sub(r"(\d{1,2}):(\d{2})", time_to_words, text)


def _normalize_money(text):
    def money_to_words(match):
        currency = match.group(1)
        amount = int(match.group(2).replace(".", ""))
        currency_text = {
            "R$": "reais" if amount > 1 else "real",
            "$": "dólares" if amount > 1 else "dólar",
            "€": "euros",
            "£": "libras",
        }.get(currency, "reais")

        amount_text = num2words(amount, lang="pt")
        return f"{amount_text} {currency_text}"

    # Regular expressions for different currency formats
    text = re.sub(r"(R\$|€|£|\$) (\d+[\.\d]*)", money_to_words, text)
    text = re.sub(r"(R\$|€|£|\$)(\d+[\.\d]*)", money_to_words, text)
    text = re.sub(r"R\$ (\d+[\.\d]*),(\d{2})", money_to_words, text)
    return text


def _normalize_numbers(text):
    return re.sub(r"\b\d+\b", lambda x: num2words(x.group(), lang="pt"), text)


def _normalize_abbreviations(text):
    for regex, substitution in abbreviations:
        text = regex.sub(substitution, text)
    return text


def _normalize_am_pm_times(text):
    def am_pm_to_words(match):
        hours = int(match.group(1))
        period = match.group(2).lower()
        if period == "pm" and hours != 12:
            hours += 12
        elif period == "am" and hours == 12:
            hours = 0
        hours_text = num2words(hours, lang="pt", to="cardinal")
        return f"{hours_text} horas"

    return re.sub(r"(\d{1,2})(am|pm)", am_pm_to_words, text)


def _normalize_numbers_with_letters(text):
    return re.sub(
        r"(\d+)([a-zA-Z]+)",
        lambda m: f"{num2words(m.group(1), lang='pt')} {m.group(2)}",
        text,
    )



