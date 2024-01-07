import json
import os
# jsonlist = ['00001.json','14919.json','20586.json']#os.listdir(path)
from pathlib import Path
### xml to json
# for xmlfile in xmllist:
#     with open(Path("../Annotations/")/xmlfile,encoding ="UTF-8") as xmlf:
#         parsed_item = xmltodict.parse(xmlf.read())
#         jsonitem    =  json.dumps(parsed_item,ensure_ascii=False)
#         with open(f"{xmlfile.split('.')[0]}.json","w",encoding = "UTF-8") as json_file:
#             json_file.write(json.dumps(parsed_item))
#             json_file.close()

### merge json
path = Path("JsonFiles")
jsonlist = os.listdir(path) # all json files
reformattedlist = []
trainidx = []
testidx  = []
validx   = []
with open("train.txt",'r') as trainf:
    for i in trainf.readlines():
        i = i.strip()
        trainidx.append(i.zfill(5))
with open("test.txt",'r') as testf:
    for i in testf.readlines():
        i = i.strip()
        testidx.append(i.zfill(5))
with open("val.txt",'r') as valf:
    for i in valf.readlines():
        i = i.strip()
        validx.append(i.zfill(5))
train_dict = []#dict()
test_dict  = []#dict()
val_dict   = []#dict()
for json_file in jsonlist:
    with open(path / json_file) as jf:
        sample = json.load(jf)
        objs = sample['annotation']['object']
        if type(objs)==dict:# only one description
            sampledict = dict()
            sampledict["height"] = int(sample['annotation']['size']['height'])
            sampledict["width"]  = int(sample['annotation']['size']['width'])
            sampledict["image_id"] = int(sample['annotation']['filename'].split('.')[0])
            sampledict["expressions"] = [objs['description']]
            x1= int(objs['bndbox']['xmin'])
            y1= int(objs['bndbox']['ymin'])
            x2= int(objs['bndbox']['xmax'])
            y2= int(objs['bndbox']['ymax'])
            sampledict["bbox"] = [x1,y1,x2,y2]
            if json_file.split('.')[0] in trainidx:
                train_dict.append(sampledict)
            elif json_file.split('.')[0] in testidx:
                test_dict.append(sampledict)
            elif json_file.split('.')[0] in validx:
                val_dict.append(sampledict)
        else: # if obj is multiple
            for obj in objs:

                sampledict = dict()
                sampledict["height"] = int(sample['annotation']['size']['height'])
                sampledict["width"]  = int(sample['annotation']['size']['width'])
                sampledict["image_id"] = int(sample['annotation']['filename'].split('.')[0])
                sampledict["expressions"] = [obj['description']]
                x1= int(obj['bndbox']['xmin'])
                y1= int(obj['bndbox']['ymin'])
                x2= int(obj['bndbox']['xmax'])
                y2= int(obj['bndbox']['ymax'])
                sampledict["bbox"] = [x1,y1,x2,y2]
                if json_file.split('.')[0] in trainidx:
                    train_dict.append(sampledict)
                elif json_file.split('.')[0] in testidx:
                    test_dict.append(sampledict)
                elif json_file.split('.')[0] in validx:
                    val_dict.append(sampledict)
instance_dict = {"train": train_dict,"val": val_dict,"test": test_dict}

with open("instances.json",'w') as insf:
    json.dump(instance_dict,insf)
print(f"train len{len(train_dict)}, test len{len(test_dict)}, val len {len(val_dict)}")
