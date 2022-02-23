import random
from summa.summarizer import summarize
from sumy.summarizers.kl import KLSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

AVG_SUMMARY_LEN = 10 #TODO: Should be average number of words among all summaries

def adjust_summary_len(summary):
    '''For the sentence which causes the summary to exceed the budget, we keep
    or discard the full sentence depending on which resulting summary is closer
    to the budgeted length.
    '''
    tokenized = summary.split('. ')
    total_len = len(summary.split())
    while len(tokenized) > 1 and total_len > AVG_SUMMARY_LEN:
        if abs(AVG_SUMMARY_LEN - total_len) > abs(AVG_SUMMARY_LEN - (total_len - len(tokenized[-1].split()))):
            total_len -= len(tokenized[-1].split())
            tokenized.pop()

    return (' ').join([str(sentence) for sentence in tokenized])


def text_rank(text):
    '''Harnesses the PageRank algorithm to choose the sentences with the
    highest similarity scores to the original document.

    Source: https://github.com/summanlp/textrank
    '''
    summary = summarize(text)
    if len(summary.split()) > AVG_SUMMARY_LEN:
        summary = adjust_summary_len(summary)
    return summary


def kl_sum(text):
    '''Greedily selects the sentences that minimize the Kullback-Lieber (KL) divergence
    between the original text and proposed summary.

    Sources: http://nlp.cs.berkeley.edu/pubs/Haghighi-Vanderwende_2009_Summarization_paper.pdf
             https://pypi.org/project/sumy/
    '''
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = KLSummarizer()

    summary = summarizer(parser.document, 1)
    for i in range(2, len(text.split('. '))):
        prev_len = len((' ').join([str(sentence) for sentence in summary]).split())
        new_summary = summarizer(parser.document, i)
        new_len = len((' ').join([str(sentence) for sentence in new_summary]).split())
        if abs(AVG_SUMMARY_LEN - new_len) >= abs(AVG_SUMMARY_LEN - new_len):
            break
        else:
            summary = new_summary

    return (' ').join([str(sentence) for sentence in summary])


def lead_one(text):
    '''Select the first sentence of the original text as the summary.
    '''
    return text.split('. ')[0]


def lead_k(text):
    '''
    Selects the first k sentences until a word limit is satisfied.
    '''
    tokenized = text.split('. ')

    summary = [tokenized[0]]
    total_len = len(tokenized[0].split())

    for i in range(1, len(tokenized)):
        len_i = len(tokenized[i].split())
        if abs(AVG_SUMMARY_LEN - total_len) > abs(AVG_SUMMARY_LEN - (total_len+len_i)):
            summary.append(tokenized[i])
            total_len += len_i
        else:
            break

    return ('. ').join(summary)


def random_k(text):
    '''Selects a random sentence until a word limit is satisfied. For this baseline,
    the reported numbers are an average of 10 runs on the entire dataset.
    '''
    tokenized = text.split('. ')

    random_shuffle = list(range(len(tokenized)))
    random.shuffle(random_shuffle)

    i = random_shuffle.pop()
    summary = [tokenized[i]]
    total_len = len(tokenized[i].split())

    while random_shuffle:
        i = random_shuffle.pop()
        len_i = len(tokenized[i].split())
        if abs(AVG_SUMMARY_LEN - total_len) > abs(AVG_SUMMARY_LEN - (total_len+len_i)):
            summary.append(tokenized[i])
            total_len += len_i
        else:
            break

    return ('. ').join(summary)


def bart_no_finetuning(text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenized_input = tokenizer(text, return_tensors="pt")
    outputs = model(**tokenized_input)


def main():
    original_text = "welcome to the pok\u00e9mon go video game services which are accessible via the niantic inc niantic mobile device application the app. to make these pok\u00e9mon go terms of service the terms easier to read our video game services the app and our websites located at http pokemongo nianticlabs com and http www pokemongolive com the site are collectively called the services. please read carefully these terms our trainer guidelines and our privacy policy because they govern your use of our services."
    reference_summary = "hi."

    test = """Automatic summarization is the process of reducing a text document with a \
    computer program in order to create a summary that retains the most important points \
    of the original document. As the problem of information overload has grown, and as \
    the quantity of data has increased, so has interest in automatic summarization. \
    Technologies that can make a coherent summary take into account variables such as \
    length, writing style and syntax. An example of the use of summarization technology \
    is search engines such as Google. Document summarization is another."""

    print("TEXT RANK")
    print(text_rank(test))
    print("KL SUM")
    print(kl_sum(test))
    print("LEAD ONE")
    print(lead_one(original_text))
    print("LEAD K")
    print(lead_k(original_text))
    print("RANDOM K")
    print(random_k(original_text))

    #TODO: ROUGE scores


if __name__ == "__main__":
    main()
