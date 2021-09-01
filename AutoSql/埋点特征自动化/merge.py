#!/usr/bin/env python
import sys
startsig = '$appstart'
endsig = '$append'
lastkey = None
threshold = 120*1000
lastline = None
for i,line in enumerate(sys.stdin): 
    line = line.strip().split("\t")
    key = line[0]
    eventid = line[1]
    timestamp = line[2]
    if lastkey!=key:
        lastkey = None
        lastline= None
        set_id = 0 
    if eventid!=endsig and eventid!=startsig:
        print('\t'.join(line+[str(set_id)]))
    elif eventid==endsig:
        if lastkey is None:
            print('\t'.join(line+[str(set_id)]))
        elif (float(lastline[2])-float(timestamp))>threshold:
            print('\t'.join(lastline+[str(set_id)]))
            set_id+=1
            print('\t'.join(line+[str(set_id)]))   
    lastline = line
    lastkey = key
    
