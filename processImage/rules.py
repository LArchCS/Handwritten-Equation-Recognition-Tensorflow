'''
This is our label rules and back_rules for each symbol in our train and predict file 
'''

class Rules():
    def __init__(self):
        self.sy = ['dots', 'tan', ')', '(', '+', '-', 'sqrt', '1', '0', '3', '2', '4', '6', 'mul', 'pi', '=', 'sin', 'pm', 'A',
        'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'div']

    def getrules(self):
        rules = {}
        lst = [float(0)] * 40 #40 buckets for each symbol
        for i in range(0,len(self.sy)):
            lst[i] = float(1) # filling each symbol in a unique box
            rules[self.sy[i]] = lst[:] # mapping a symbol to a list of [0,0,1,...0,0,0]
            lst[i] = float(0)
        #later we can do some merge rules: 0 and o, frac and bar and -, x and mul
        rules['o'] = rules['0'] # make o's label 0 and frac, bar's label -, mul's label x
        rules['frac'] = rules['-']
        rules['bar'] = rules['-']
        rules['mul'] = rules['x']
        return rules

    def getbrules(self):
        brules = {}
        for i in range(0,len(self.sy)):
            brules[i] = self.sy[i]
        return brules # note that 0 and o, frac and bar and -, x and mul are in the same bucket
