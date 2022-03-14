# text-mining
Classification of articles from the Reuters dataset


# Text Mining: Classification of articles from the Reuters dataset

 The Forex market is a
global marketplace where most of national currencies are traded and it does not
differ much from the stock market: in the latter, investors try to take advantage of
the fluctuations of stock prices; in the former, traders try to exploit the fast varia-
tions of currencies’ exchange rates. There are, however, few differences between the
currency market and the stock market.

A unique feature of the Forex market is that it is open 24 hours a day, 5 days a week.
As a consequence, retail investors may find it hard to keep up with the fluctuations
in exchange rates caused by the constant stream of information from all over the
world. A software able to promptly read and understand news articles and quickly
perform appropriate operations in the Forex market, thus appears handy.

# About this project:

In this project, we will classify text documents based on their topic. In particular, our model will
inform the user about whether or not a certain article is related to the Forex market
(i.e topic = Money-FX) with an accuracy score of 97.8%



# Dataset:

* [Dataset link](https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection)

The data-set we used to train our program and to subsequently perform predictions
and test the accuracy is called Reuters 21758. This corpus of documents is made up
of 22 files under .sgm format. The first 21 files contain 1000 articles each and the
22nd contains the remaining 758 articles. Each article has been read and labeled
with one or more topics by a human reader.

The collection of 21758
documents contains articles without labeled topics and/or without body. The arti-
cles in each file are delimited by the tags <reuter>, </reuter>. Similarly, the articles
are themselves divided into sections by the respective tags such as <date>, </date>;
<places>, </places> or <body>, </body>. Throughout the project, we were mostly
interested in isolating the tags <topics>, </topics> and <body>, </body> for each
article.


One of the fundamental aspects at the core of the prediction process is to make
use of the word-frequency of a certain article in order to make predictions about its
topic. We decide to use word-plots to obtain an interesting view on the most used
words. For the visual representation of the text data we use Wordcloud( The Python
Graphs Gallery). Wordcloud shows a list of words with the importance of each word
being aligned with the font size. This visual representation is useful to perceive the
most prominent terms quickly. The first wordcloud of the articles having "money-
fx" as topic. The second wordcloud is of the articles
that are not about "money-fx".

Wordcloud of money-fx articles

![5](https://user-images.githubusercontent.com/20649715/158151545-a0c1453f-23ca-48c5-905e-0d876e568b05.jpeg)

Wordcloud of non-money-fx (financial and economic articles) articles

![6](https://user-images.githubusercontent.com/20649715/158153418-772f70af-da38-4942-9bb2-14c6197f7d06.jpeg)




# Article processing


Article before

```html
<reuters cgisplit="TRAINING-SET" lewissplit="TRAIN" newid="9" oldid="5552" topics="YES">
<date>26-FEB-1987 15:17:11.20</date>
<topics><d>earn</d></topics>
<places><d>usa</d></places>
<people></people>
<orgs></orgs>
<exchanges></exchanges>
<companies></companies>
<unknown>
F
f0762reute
r f BC-CHAMPION-PRODUCTS-&lt;CH 02-26 0067</unknown>
<text>
<title>CHAMPION PRODUCTS &lt;CH&gt; APPROVES STOCK SPLIT</title>
<dateline> ROCHESTER, N.Y., Feb 26 - </dateline><body>Champion Products Inc said its
board of directors approved a two-for-one stock split of its
common shares for shareholders of record as of April 1, 1987.
The company also said its board voted to recommend to
shareholders at the annual meeting April 23 an increase in the
authorized capital stock from five mln to 25 mln shares.
Reuter
</body></text>
</reuters>
```

```html
Article after
[“earn”,
“champion product inc board director approv twoforon stock split common share sharehold
record april compani also board vote recommend sharehold annual meet april increas author
capit stock five mln mln share”]
```

During preprocessing the articles, the common stopwords were removed and new stopwords identified with the help of the wordclouds. Considering the financial nature of articles, all numbers were replaced with the word "num". Stemming was performed - the process of reducing a certain word to its basic root (or stem) in order to make the dataset more compact.

#Model

*readme currently under construction*

