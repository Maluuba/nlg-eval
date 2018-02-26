from nlgeval import NLGEval

def test_oo_api():
    with open("examples/hyp.txt") as f:
        hyp = f.readlines()
    with open("examples/ref1.txt") as f:
        ref1 = f.readlines()
    with open("examples/ref2.txt") as f:
        ref2 = f.readlines()

    nlge = NLGEval()
    res = nlge.evaluate(ref1, hyp[0])
    import ipdb; ipdb.set_trace();
    res = nlge.evaluate(ref2, hyp[1])
