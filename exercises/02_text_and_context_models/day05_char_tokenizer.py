def build_char_tokenizer(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return [stoi[ch] for ch in text]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


def main():
    text = "hello llm\nhello infra"
    stoi, itos = build_char_tokenizer(text)

    ids = encode(text, stoi)
    decoded = decode(ids, itos)

    print("text:")
    print(repr(text))

    print("\nchars:")
    print(list(stoi.keys()))

    print("\nstoi:")
    print(stoi)

    print("\nitos:")
    print(itos)

    print("\nvocab_size:", len(stoi))

    print("\nencoded ids:")
    print(ids)

    print("\ndecoded text:")
    print(repr(decoded))

    print("\nround trip works:", decoded == text)

    print("\nMeaning:")
    print("encode converts text into token ids.")
    print("decode converts token ids back into text.")


if __name__ == "__main__":
    main()
