import json
import xmltodict
import os
from pathlib import Path
path = Path("Annotation/")
dstfile = "instances.json"
def xml2dict(path):
    xmllist = os.listdir(path)
    for xmlfile in xmllist:
        with open( path / xmlfile,encoding ="UTF-8") as xmlf:
            parsed_item = xmltodict.parse(xmlf.read())
            jsonitem    =  json.dumps(parsed_item,ensure_ascii=False)
            with open(f"{xmlfile.split('.')[0]}.json","w",encoding = "UTF-8") as json_file:
                json_file.write(parsed_item)
                json_file.close()

def mergejson(path):
    jsonlist = ['00001.json','14919.json','20586.json']#os.listdir(path)
    jsondict = dict()
    for json_file in  jsonlist:
        with open(path / json_file,'r',encoding="UTF-8") as jf:
            sample = json.load(jf)
            jsondict['train'].update(sample)
            print(jsondict)