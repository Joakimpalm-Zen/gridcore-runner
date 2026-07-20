#!/usr/bin/env python3
"""Generate the tokenizer differential corpus.

The 721-string corpus used for every model in the lineup was previously
generated ad hoc and never committed, so no later agent could reproduce a
number. This script rebuilds it deterministically: no RNG, no network, no
dependencies. Run it to regenerate tests/fixtures/tokenizer-corpus.txt.

Encoding: one string per line, each line a JSON string literal, so leading and
trailing whitespace, newlines and tabs survive the round trip exactly. Those
are the cases that actually separate tokenizers -- the only known divergence in
the whole lineup (Mistral, 2 strings) is a leading-space case.

The corpus is validated by reproducing the known baseline: 0 divergences for
Llama-3.x, Qwen2.5/3, gemma-3, SmolLM2 and Phi-3.5, and exactly the 2
leading-whitespace divergences for Mistral-v0.3. A corpus that does not
reproduce Mistral's 2 is not exercising the boundary and should not be trusted.
"""

import json
import sys

TARGET = 721


def corpus():
    out = []

    def add(*items):
        out.extend(items)

    # --- plain ASCII words and sentences ------------------------------------
    add(
        "Hello world",
        "hello world",
        "HELLO WORLD",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "the quick brown fox jumps over the lazy dog",
        "A",
        "a",
        "I",
        "an apple a day",
        "Supercalifragilisticexpialidocious",
        "antidisestablishmentarianism",
        "pneumonoultramicroscopicsilicovolcanoconiosis",
        "llama",
        "Llama",
        "LLAMA",
        "tokenizer",
        "pre-tokenizer",
        "byte-level BPE",
        "SentencePiece",
    )

    # --- the empty string and single characters -----------------------------
    add("")
    for c in " \t\n\r!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
        add(c)
    for c in "abcxyzABCXYZ0123456789":
        add(c)

    # --- leading whitespace: the Metaspace prepend_scheme boundary ----------
    # This block is what catches Mistral. Keep it.
    for base in ["hello", "Hello world", "1", "!", "the"]:
        add(" " + base)
        add("  " + base)
        add("   " + base)
        add("\t" + base)
        add("\n" + base)
        add("\r\n" + base)
        add(" \t " + base)

    # --- trailing whitespace -------------------------------------------------
    for base in ["hello", "Hello world", "1", "!"]:
        add(base + " ")
        add(base + "  ")
        add(base + "\t")
        add(base + "\n")
        add(base + "\n\n")
        add(base + " \n ")

    # --- whitespace-only strings ---------------------------------------------
    add(
        " ",
        "  ",
        "   ",
        "    ",
        "        ",
        "\t",
        "\t\t",
        "\n",
        "\n\n",
        "\n\n\n",
        "\r",
        "\r\n",
        "\r\n\r\n",
        " \n",
        "\n ",
        " \n ",
        "\t\n\t",
        " " * 16,
        " " * 32,
        "\n" * 8,
    )

    # --- digits: \p{N} vs \p{N}{1,3} split rules -----------------------------
    add(
        "0", "1", "12", "123", "1234", "12345", "123456", "1234567",
        "12345678", "123456789", "1234567890", "00", "007", "0000",
        "3.14159", "2.71828", "-1", "+1", "1,000", "1,000,000",
        "1_000_000", "0x1F", "0b1010", "0o777", "1e10", "1e-10",
        "1.5e+3", "42", " 42", "42 ", "v1.2.3", "2026-07-20",
        "12:34:56", "99.9%", "$1,234.56", "1/2", "3-4", "5*6",
    )
    for n in range(0, 10):
        add(str(n) * 3)

    # --- contractions: the (?i:'s|'t|'re|'ve|'m|'ll|'d) rule -----------------
    for c in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
              "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D"]:
        add("it" + c)
        add("It" + c)
    add(
        "don't", "won't", "can't", "shouldn't", "y'all", "o'clock",
        "rock'n'roll", "'tis", "'twas", "I'm", "we're", "they've",
        "he'd", "she'll", "it's", "IT'S", "It'S",
        "’s", "don’t", "it’s",  # curly apostrophe
    )

    # --- punctuation runs -----------------------------------------------------
    add(
        "...", "....", "!!!", "???", "?!", "!?", "--", "---", "----",
        "==", "===", "====", "->", "=>", "<-", "<=", ">=", "!=", "==",
        "::", ":::", "//", "///", "/*", "*/", "#", "##", "###", "####",
        "**", "***", "``", "```", '"""', "''", "'''", "<<", ">>", "<<<",
        "&&", "||", "|", "^", "~", "%", "@", "$", "\\", "\\\\", "\\n",
        "\\t", "()", "[]", "{}", "<>", "(){}[]", "((()))",
    )

    # --- newline / slash interactions (Apertus's [\r\n/]* vs [\r\n]*) --------
    add(
        "a/b", "a//b", "a/\nb", "//\n", "/\n", "///\n", "*/\n",
        "path/to/file", "/usr/local/bin", "./relative/path",
        "../parent", "C:\\Windows\\System32", "https://example.com/a/b",
        "a/b/c/d/e/f", "//comment\n", "/**/\n", "#!/bin/sh\n",
        "\n/", "\n//", "/\r\n", "//\r\n",
    )

    # --- code ------------------------------------------------------------------
    add(
        "def f(x):\n    return x + 1\n",
        "int main(void) { return 0; }",
        "#include <stdio.h>\n",
        "if (a == b) {\n\tprintf(\"eq\\n\");\n}\n",
        "for i in range(10):\n    print(i)\n",
        "const x = () => ({ a: 1 });",
        "SELECT * FROM t WHERE id = 1;",
        "{\"key\": \"value\", \"n\": 123}",
        "<html><body><p>hi</p></body></html>",
        "x = [1, 2, 3]; y = x[0]",
        "std::vector<int> v;",
        "let mut s = String::new();",
        "func main() {\n\tfmt.Println(\"hi\")\n}",
        "public static void main(String[] args) {}",
        "SELECT\n  a,\n  b\nFROM t\n",
        "$ ls -la /tmp\n",
        "git commit -m \"fix: thing\"\n",
        "npm install --save-dev typescript",
        "curl -sSL https://example.com | sh",
        "```python\nprint('hi')\n```",
    )

    # --- indentation (repeated-space merges) ----------------------------------
    for n in [1, 2, 3, 4, 6, 8, 12, 16]:
        add(" " * n + "indented")
        add("\n" + " " * n + "indented")
    for n in [1, 2, 3, 4]:
        add("\t" * n + "tabbed")

    # --- accented Latin / European languages ----------------------------------
    add(
        "café", "naïve", "résumé", "crème brûlée",
        "über", "Straße", "grüßen", "München",
        "español", "niño", "mañana", "¿Cómo estás?",
        "français", "être", "déjà vu", "à la carte",
        "português", "São Paulo", "coração",
        "italiano", "perché", "città",
        "Ångström", "København", "Reykjavík",
        "Zurich", "Zürich", "Genève", "Neuchâtel",
        "polski", "łódź", "żabka",
        "česky", "Příbram", "žluoučký",
        "magyar", "Óbuda", "Türkçe", "İstanbul",
        "română", "București",
    )

    # --- non-Latin scripts ------------------------------------------------------
    add(
        "你好", "你好，世界",
        "中文测试", "繁體中文",
        "こんにちは", "日本語",
        "カタカナ", "ひらがな",
        "안녕하세요", "한국어",
        "Привет", "русский",
        "Γεια σου", "ελληνικά",
        "שלום", "עברית",
        "مرحبا", "العربية",
        "नमस्ते", "हिन्दी",
        "สวัสดี", "ไทย",
        "ሰላም", "አማርኛ",
        "բարեւ",
        "გამარჯობა",
        "Tiếng Việt", "Български",
    )

    # --- emoji and symbols -------------------------------------------------------
    add(
        "\U0001f600", "\U0001f680", "\U0001f4a1", "\U0001f1e8\U0001f1ed",
        "\U0001f469‍\U0001f4bb", "\U0001f3f3️‍\U0001f308",
        "\U0001f44d\U0001f3fd", "hi \U0001f600 there",
        "\U0001f600\U0001f600\U0001f600",
        "❤️", "✅", "❌", "⚠️",
        "→", "⇒", "∈", "∑", "∞", "≈",
        "±", "×", "÷", "≠", "≤", "≥",
        "αβγ", "∑(xᵢ)", "∫ f(x) dx",
        "▁", "▁▁", "▁hello",  # the SPM metaspace char itself
        " ", "a b",                      # non-breaking space
        "​", "a​b",                      # zero-width space
        "﻿", "﻿hello",                   # BOM
    )

    # --- special-token-shaped strings (parse_special boundary) ------------------
    add(
        "<s>", "</s>", "<unk>", "<pad>", "<|endoftext|>", "<|im_start|>",
        "<|im_end|>", "<|end|>", "<|user|>", "<|assistant|>", "<|system|>",
        "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "<|begin_of_text|>",
        "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
        "<start_of_turn>", "<end_of_turn>", "[TOOL_CALLS]",
        "<|assistant_end|>", "<|inner_prefix|>", "<|tools_prefix|>",
        "not a <s> real tag", "a<s>b", "<s", "s>", "<|", "|>",
        "<<<>>>", "< s >", "<S>",
    )

    # --- mixed case and camelCase ------------------------------------------------
    add(
        "camelCase", "PascalCase", "snake_case", "kebab-case",
        "SCREAMING_SNAKE", "mixedUP", "aB", "Ab", "AB", "aBcDeF",
        "XMLHttpRequest", "getHTTPResponse", "IOError", "iOS", "macOS",
        "eBay", "iPhone", "GDPR", "EU AI Act", "ISO/IEC 42001",
    )

    # --- URLs, emails, identifiers -------------------------------------------------
    add(
        "https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509",
        "http://localhost:8080/v1/chat/completions",
        "user@example.com", "first.last+tag@sub.domain.co.uk",
        "192.168.1.1", "::1", "2001:db8::1",
        "550e8400-e29b-41d4-a716-446655440000",
        "sk-abc123XYZ", "#hashtag", "@mention", "u/user", "r/subreddit",
    )

    # --- long / repetitive ----------------------------------------------------------
    add(
        "a" * 64,
        "ab" * 32,
        "word " * 20,
        "The quick brown fox. " * 8,
        "中" * 32,
        "1" * 40,
        "=" * 40,
        " " * 8 + "x" + " " * 8,
    )

    # --- prose paragraphs -------------------------------------------------------------
    add(
        "Apertus is a fully open multilingual large language model developed by "
        "the Swiss AI Initiative, released with open weights, open training "
        "data and open training recipes.",
        "Runner loads GGUF weights from disk, binds INADDR_LOOPBACK, and makes "
        "no outbound network call of any kind.",
        "In der Schweiz gesprochene Sprachen sind Deutsch, Französisch, "
        "Italienisch und Rätoromanisch.",
        "Le modèle est publié sous licence Apache 2.0, ce qui autorise "
        "l'usage commercial.",
        "Il modello è rilasciato con licenza Apache 2.0.",
        "Regulation (EU) 2024/1689 lays down harmonised rules on artificial "
        "intelligence.",
    )

    # --- deterministic filler: systematic short strings ---------------------------------
    # Pads to TARGET without introducing randomness. Each is still a real
    # tokenizer boundary case: a separator between two word-ish fragments.
    seps = [" ", "  ", "\t", "\n", "-", "_", ".", ",", ":", ";", "/", "\\",
            "|", "+", "=", "(", ")", "'", '"', "*"]
    lefts = ["a", "A", "the", "The", "1", "x1", "中", "café"]
    rights = ["b", "B", "end", "End", "2", "y2", "文", "naïve"]
    i = 0
    while len(out) < TARGET:
        s = seps[i % len(seps)]
        l = lefts[(i // len(seps)) % len(lefts)]
        r = rights[(i // (len(seps) * len(lefts))) % len(rights)]
        cand = l + s + r
        if cand not in out:
            out.append(cand)
        i += 1
        if i > 100000:
            break

    return out[:TARGET]


def main():
    c = corpus()
    if len(c) != TARGET:
        print(f"error: corpus has {len(c)} strings, expected {TARGET}",
              file=sys.stderr)
        return 1
    dst = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/tokenizer-corpus.txt"
    with open(dst, "w", encoding="utf-8") as f:
        for s in c:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {len(c)} strings to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
