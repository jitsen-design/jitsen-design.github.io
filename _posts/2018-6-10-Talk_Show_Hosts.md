---
layout: post
title: NMF and LDA to analyze Difference in Topicality and Frequency of Tweets among Top Talkshow Hosts
---

This project explores Twitter activity of prominent talk show hosts: Namely, Bill Maher, Bill O'Reilly, Sean Hannity and John Oliver. It also uses Non-Negative Matrix Factorization (NMF) and Latent Dirichlet Allocation (LDA) to look at top themes within those tweets.

First, let's look at the how frequently each show host tweets. It seems that Sean Hannity is consistently on top, followed by Bill O'Reilly. It is safe to say that the Conservative hosts are more active on Twitter. To be fair, John Oliver has not had his show for a very long time, and his activity on Twitter is limited.

![Number of Tweets](../images/tweets.png?raw=true)

We can also look at how often, each tweet is retweeted. It is clear that the flurry of activity on Twitter in terms of retweets from all hosts is recent. Perhaps this this due to the fact that people only recently began to realize how potent of a tool Twitter really is.

![Number of Retweets](../images/retweets.png?raw=true)

Finally we can analyze trends in retweets per tweet for a talk show host, but dividing the sum of retweets for a given year by the number of total tweets for that year. We see an exponential upward trend here. It is also noticieable that the Liberal hosts enjoy a higher % of retweets, possibly due to their relatively low frequency of tweets.

![Average Retweets per Tweet](../images/relative_retweets.png?raw=true)

Finally we can look at themes or topicality of tweets by using Non-Negative Matrix Factorization and Latent Dirichlet Allocation.

Matrix factorization returns features from a dataset ranked according to how much variance each feature captures. It uses singular value decomposition (SVD) to split a given dataset into feature vectors. Non-Negative Matrix Factorization essentially uses singular value decomposition, but preserves only non-negative values in terms of word weights (probabilities). The word weights here  are extracted using Term Frequency / Inverse Document Frequency vectorization (TFIDF). We can then get pre-defined features (five  in this case), which represent the top dimensions of the TFIDF vectors. Let's look at the top five topics (dimensions) for both Bill Maher and Sean Hannity, limited to five words for each topic.


#### Bill Maher

##### NMF
* Topic 0: video realtime clip via newrules
* Topic 1: like trump one get say
* Topic 2: time real hbo video clip
* Topic 3: tonight show live nbc leno
* Topic 4: comedy ticket stand sale today

##### LDA
* Topic 0: dont obama trump get president
* Topic 1: like cuz really going america
* Topic 2: clip video time tonight realtime
* Topic 3: im like one day get
* Topic 4: year new say good never

#### Sean Hannity

##### NMF
* Topic 0: hannity tonight next start joined
* Topic 1: lnyhbt yes zeller get army
* Topic 2: thanks welcome great stick stay
* Topic 3: ha zeller yes good michaelmbsc
* Topic 4: thx love great joe go

##### LDA
* Topic 0: lnyhbt trump people obama love
* Topic 1: hannity next new join talk
* Topic 2: think yes last take hey
* Topic 3: great get thanks zeller good
* Topic 4: hannity tonight day today radio

There doesn't seem to be very much to decipher in terms of clear partisanship between the two. We can possiblly say that Sean Hannity references other Twitter users (lnyhbt, zeller) more often than Bill Maher. Perhaps we can decipher more by analyzing Bill O'Reilly and John Oliver.

#### Bill O'Reilly

##### NMF
* Topic 0: talking point memo transcript rush
* Topic 1: video watch obama clip obamacare
* Topic 2: factor tonight clip well trump
* Topic 3: quiz news latest history american
* Topic 4: column new crossword read online

##### LDA
* Topic 0: detail clinton hillary story party
* Topic 1: tonight factor well daily briefing
* Topic 2: point talking watch president obama
* Topic 3: video watch column new latest
* Topic 4: factor clip watch quiz history

#### John Oliver

##### NMF
* Topic 0: piece last week trump watching
* Topic 1: thanks huge voice helping help
* Topic 2: tonight new back show week
* Topic 3: night last upcoming sunday enjoy
* Topic 4: many thanks great helping help

##### LDA
* Topic 0: thanks huge many great helping
* Topic 1: tonight show new watching dont
* Topic 2: im tomorrow going year part
* Topic 3: last piece night week thank
* Topic 4: back like time sunday one

The results for this comparison is disappointing as well. We are unable to see distinctions between these two hosts as well. Finally, let's take a look at  Word2Vec vectorization of each host plotted on t-SNE with Pincipal Component Analysis (PCA) dimension reduction. PCA with t-SNE allows us th plot 100 dimensional world vectors to be plotted on a 2D visual

![Word Embeddings for all hosts](../images/bokeh.png?raw=true)

The word plots show the difference in scale of activity between the Liberal and Conservative talkshow hosts. You can also see that word relationships are more sparse for John Oliver than for Bill Maher. This is natural, given that John Oliver covers a broader varierty of topics in his weekly show than Bill Maher does in his nightly offering.
