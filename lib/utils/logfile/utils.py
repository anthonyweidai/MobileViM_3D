from ..system import colourText


def listPrinter(ListName, ListValues, LineNum=None, **kwargs):
    Len1 = len(ListName)
    Len2 = len(ListValues)
    assert Len1 == Len2, "Got not consistent len %d and %d" % (Len1, Len2)
    
    if LineNum == None:
        LineNum = 5
    
    Remainder = Len1 % LineNum
    if Remainder > 0 and Remainder <3:
        LineNum -= 1
        Remainder = Len1 % LineNum
        if Remainder > 0 and Remainder < 2:
            LineNum += 2
            Remainder = Len1 % LineNum
            if Remainder > 0 and Remainder < 3:
                LineNum -= 2
                  
    PrintStr = ""
    for Idx in range(Len1):
        if Idx > 0 and Idx % LineNum == 0:
            PrintStr += "\n"
        
        Value = ListValues[Idx]
        if isinstance(Value, float):
            if Value > 1:
                Value = round(Value, 2)
            else:
                Value = round(Value, 4)
        
        PrintStr += "%s: %s, " % (ListName[Idx], colourText(str(Value), **kwargs))
        # %r for bool printing
    print(PrintStr)
