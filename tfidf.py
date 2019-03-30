#this is aimed at building a term frequency inverse document frrequency vectorizer.
def term_frequency(document):
    """
     document: this is to compute the term frequency of the document posted
    """

    reviewtfdict = {}
    # counts the number of times each item appears
    for word in document:
        if word in reviewtfdict:
            reviewtfdict[word] +=  1
        else:
            reviewtfdict[word]= 1

    #computes the term frequency
    for word in reviewtfdict:
        reviewtfdict[word]= reviewtfdict[word]/ len(document)
    return reviewtfdict


document= ['aromas', 'include', 'tropical', 'fruit', 'broom','broom','brimstone', 'brimstone',
  'and', 'dried', 'herb', 'the', 'palate', "isn't", 'overly','sage',
  'expressive', 'offering', 'unripened', 'apple', 'citrus',
  'and', 'dried', 'sage', 'alongside', 'brisk', 'acidity']


def computecountdict():
    
    """ Returns a dictionary whose keys are all the unique words in
    the dataset and whose values count the number of reviews in which
    the word appears.
    """
    countdict={}
    for word in document:
        if word in countdict:
            countdict[word] += 1
        else:
            countdict[word]= 1

    return countdict
countdict= computecountdict()

import math

def computeidfdict():
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    
    idfdict= {}
    for word in countdict:
        idfdict[word]= math.log(len(document) / countdict[word])
    return idfdict
idfdict= computeidfdict()


def computetfidfdict(reviewtfdict):
    tfidfdict= {}
    #for
    for word in tfidfdict:
        tfidfdict[word]= reviewtfdict[word] * idfdict[word]
    return tfidfdict
tfidfdict = [computetfidfdict(document) for document in countdict]
    




