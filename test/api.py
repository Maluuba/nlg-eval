from nlgeval import NLGEval

def test_oo_api():
    with open("examples/hyp.txt") as f:
        hyp = f.readlines()
    with open("examples/ref1.txt") as f:
        ref1 = f.readlines()
    with open("examples/ref2.txt") as f:
        ref2 = f.readlines()

    nlge = NLGEval()
    res = nlge.evaluate([ref1[0]] + [ref2[0]], hyp[0])
    res = nlge.evaluate([ref1[1]] + [ref2[1]], hyp[1])
