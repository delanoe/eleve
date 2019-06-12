
#from eleve import MemoryStorage
#from eleve import MemoryTrie
from eleve.memory import MemoryStorage
from eleve.memory import MemoryTrie
from math import isnan


EPSILON = 0.0001

def float_equal(a, b):
    return (a != a and b != b) or abs(a - b) < EPSILON



corpus0 = [["New","York","is","New","York","and","New","York"]]
corpus0t = ["\ue02b","New","York","is","New","York","and","New","York","\ue02d"]

tokens0 = [["\ue02b"],["New"],["York"],["is"],["and"],["\ue02d"],
           ["\ue02b","New"],["New","York"],["York","is"],["is","New"],["York","and"],["and","New"],["York","\ue02d"],
           ["\ue02b","New","York"],["New","York","is"],["York","is","New"],["is","New","York"],["New","York","and"],["York","and","New"],["and","New","York"],["New","York","\ue02d"]
          ]

tokens1 = [["\ue02b"],["New"],["York"],["is"],["New"],["York"],["and"],["New"],["York"],["\ue02d"]]
tokens2 = [["\ue02b","New"],["New","York"],["York","is"],["is","New"],["New","York"],["York","and"],["and","New"],["New","York"],["York","\ue02d"]]
tokens3 = [["\ue02b","New","York"],["New","York","is"],["York","is","New"],["is","New","York"],["New","York","and"],["York","and","New"],["and","New","York"],["New","York","\ue02d"],["York","\ue02d"],["\ue02d"],[]]

#corpus1 = [["to","be","or","not","to","be","or","NOT","to","be","and"]]
#tokens1 = [["to","be"],["be","or"],["or","not"],["not","to"],["to","be"],["be","or"],["or","NOT"],["NOT","to"],["to","be"],["be","and"]]

def test_unidirection(corpus,tokens):
    storage = MemoryTrie()
    for sent in corpus:
        storage.add_sentence(sent)

    result = []

    for token in tokens:
        result.append( ( token
                       , storage.query_count(token)
                       , storage.query_entropy(token)
                       , storage.query_ev(token)
                        )
                     )
    print(result)

def test_bidirection(corpus,tokens):
    storage = MemoryStorage()

    for sent in corpus:
        storage.add_sentence(sent)

    storage.fwd.update_stats()
    storage.bwd.update_stats()
    print('fwd normalization:', storage.fwd.normalization)
    print('bwd normalization:', storage.bwd.normalization)

    print("[(token, count, entropy, ev, autonomy, fwd_entropy, fwd_ev, fwd_autonomy, bwd_entropy, bwd_ev, bwd_autonomy)]")
    for token in tokens:
        tokenRev = token[::-1]
        print(",",(" ".join(token)
              , storage.query_count(token)
              , storage.query_entropy(token)
              , storage.query_ev(token)
              , storage.query_autonomy(token)
              
              , storage.fwd.query_entropy(token)
              , storage.fwd.query_ev(token)
              , storage.fwd.query_autonomy(token)
              
              , storage.bwd.query_entropy(tokenRev)
              , storage.bwd.query_ev(tokenRev)
              , storage.bwd.query_autonomy(tokenRev)
              ))

def test_mean(ref, f, b):
    my = (f + b) / 2
    
    if float_equal(ref,my) or (isnan(ref) and isnan(my)):
        print("PASS")
    else:
        print("FAIL", 'ref:', ref, 'my:', my, 'fwd:', f, 'bwd:', b)

def test_bidirection_mean(corpus,tokens):
    storage = MemoryStorage()

    for sent in corpus:
        storage.add_sentence(sent)

    for token in tokens:
        tokenRev = token[::-1]
        test_mean(storage.query_entropy(token),
                  storage.fwd.query_entropy(token),
                  storage.bwd.query_entropy(tokenRev))
        test_mean(storage.query_ev(token),
                  storage.fwd.query_ev(token),
                  storage.bwd.query_ev(tokenRev))
        test_mean(storage.query_autonomy(token),
                  storage.fwd.query_autonomy(token),
                  storage.bwd.query_autonomy(tokenRev))

def tests_bidirection():
    test_bidirection_mean(corpus0,tokens0)
#   test_bidirection(corpus0,tokens0)
#    test_bidirection(corpus0,tokens2)
#    test_bidirection(corpus0,tokens3)

def tests_unidirection():
    print("[(token, count, entropy, ev, autonomy)]")
    test_unidirection(corpus0,tokens0)
    #test_unidirection(corpus1,tokens1)

tests_bidirection()
