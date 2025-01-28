import re
from collections import Counter

class CustomTokenizer:
    def __init__(self, vocab_size=10000, lower=True):
        self.vocab_size = vocab_size
        self.lower = lower
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}  # 特殊标记
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
    
    def fit(self, texts):
        # 文本预处理
        processed_texts = [self._preprocess(text) for text in texts]
        word_counts = Counter(word for text in processed_texts for word in text.split())
        most_common_words = word_counts.most_common(self.vocab_size - len(self.word2idx))

        # 构建词汇表
        for i, (word, _) in enumerate(most_common_words, start=len(self.word2idx)):
            self.word2idx[word] = i
            self.idx2word[i] = word
    
    def tokenize(self, text):
        # 将文本转化为 token ID 序列
        text = self._preprocess(text)  # 预处理文本
        tokens = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in text.split()]
        return [self.word2idx["<SOS>"]] + tokens + [self.word2idx["<EOS>"]]
    
    def detokenize(self, tokens):
        # 将 token ID 序列还原为文本
        sentence = " ".join([self.idx2word.get(token, "<UNK>") for token in tokens])
        # 去除标点前的空格
        sentence = re.sub(r"\s([.,!?'])", r"\1", sentence)
        return sentence

    
    def _preprocess(self, text):
        if self.lower:
            text = text.lower()
        # 在标点符号前后添加空格，使标点与单词分开
        text = re.sub(r"([.,!?'])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text)  # 去除多余空格
        return text.strip()


if __name__ == "__main__":
    # 加载数据
    with open("datasets/train/train.en", "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 初始化 tokenizer
    tokenizer = CustomTokenizer(vocab_size=10000)
    tokenizer.fit(lines)  # 使用训练数据拟合词汇表

    # 示例: 分词和编码
    sample_sentence = "Two young, White males are outside near many bushes."
    tokens = tokenizer.tokenize(sample_sentence)
    decoded_sentence = tokenizer.detokenize(tokens)

    print("Original Sentence:", sample_sentence)
    print("Token IDs:", tokens)
    print("Decoded Sentence:", decoded_sentence)
