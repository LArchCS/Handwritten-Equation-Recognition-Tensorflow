import os
import numpy
import pprint

pp = pprint.PrettyPrinter(indent=4)
eq = {}
peq = {}

def marklabel(f):
   ins = f.split('.')[0].split('_')
   if eq.has_key(ins[1]+ins[2]):
       if len(ins) > 3:
           eq[ins[1]+ins[2]][ins[4]+"_"+ins[5]+"_"+ins[6]+"_"+ins[7]] = ins[3]
   else:
       eq[ins[1]+ins[2]] = {}

def getpeq():
    pf = open(os.getcwd()+"/predictions_new.txt","r")
    for line in pf.readlines():
        ins = line.split("\t")
        if len(ins) < 4:
            t = ins[0].split(".")[0].split("_")
            eqname = t[1]+t[2]
            peq[eqname] = {}
        else:
            peq[eqname][ins[1]+"_"+ins[2]+"_"+ins[3]+"_"+ins[4]] = ins[0]

def getaccuracy():
    total = 0
    hit = 0
    bad_result = open("./unmatch.txt","w")
    all_result = open("./all_result.txt","w")
    for equa in eq.keys():
        if peq.has_key(equa):
            bad_result.write(equa)
            for pos in eq[equa].keys():
                ins = pos.split("_")
                for ppos in peq[equa].keys():
                    ins1 = ppos.split("_")
                    if  abs(int(ins[0]) - int(ins1[1])) < 6 and abs(int(ins[1]) - int(ins1[3])) < 6 and abs(int(ins[2]) - int(ins1[0])) < 6 and abs(int(ins[3]) - int(ins1[2])) < 6:
                        if eq[equa][pos] == "mul": eq[equa][pos] = "x"
                        if peq[equa][ppos] == "mul": peq[equa][ppos] = "x"
                        if eq[equa][pos] == "bar" or eq[equa][pos] == "frac": eq[equa][pos] = "-"
                        if peq[equa][ppos] == "bar" or peq[equa][ppos] == "frac": peq[equa][ppos] = "-"
                        if eq[equa][pos] == "o": eq[equa][pos] = "0"
                        if peq[equa][ppos] == "o": peq[equa][ppos] = "0"
                        if eq[equa][pos] == peq[equa][ppos]:
                              hit = hit + 1
                              all_result.write("GOOD\t"+equa+"\t"+pos+"\t"+eq[equa][pos]+"\t"+peq[equa][ppos]+"\n")
                        else:
                            all_result.write("BAD!\t"+equa+"\t"+pos+"\t"+eq[equa][pos]+"\t"+peq[equa][ppos]+"\n")
                            bad_result.write(equa+"\t"+pos+"\t"+eq[equa][pos]+"\t"+peq[equa][ppos]+"\n")
                            print equa, pos, " ", eq[equa][pos], " ", peq[equa][ppos]
                total = total + 1 
    print hit / float(total) 

def main():
    dataroot = os.getcwd() + "/data/annotated"
#     total = 0
    for f in os.listdir(dataroot):
        if f.endswith(".png"):
            marklabel(f)
#             total = total + 1
#     print total
    getpeq()
#     pp.pprint (peq)
    getaccuracy()
   
            
        

if __name__ == "__main__":
    main()
