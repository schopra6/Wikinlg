import numpy as np

def getReferencefile(referencefilepath):
    references =np.load(referencefilepath,allow_pickle=True)
    temp = [x+ ['']*4 for x in references]
    equalref = [x[0:4] for x in temp]
    res0, res1,res2,res3 = map(list, zip(*equalref))
    with open('reference0', "w") as f:
            for res in res0:
                f.write(f"{res}\n")
    with open('reference1', "w") as f:
            for res in res1:
                f.write(f"{res}\n")
    with open('reference2', "w") as f:
            for res in res2:
                f.write(f"{res}\n")
    with open('reference3', "w") as f:
            for res in res3:
                f.write(f"{res}\n")


