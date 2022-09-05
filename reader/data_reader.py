import os.path
from collections import defaultdict
from datasets import Dataset, DatasetDict

from models.ModelPath import get_data_path

def load_dataset(double_lan:str):
    data_dict = {}
    names=["train"]
    src_trg=double_lan.split(sep="_")
    for name in names:
        srctrgs=defaultdict(list)
        with open(os.path.join(get_data_path(),double_lan+"."+name),"r",encoding='utf-8') as f:
            lines=f.readlines()
            for line in lines:
                srcAndTrg=line.split(sep='\t')
                if len(srcAndTrg)==1:
                    srcAndTrg=line.split(sep=' ')
                if len(srcAndTrg)!=2:
                    print("Filter",line)
                    continue
                srctrgs[src_trg[0]].append(srcAndTrg[0])
                srctrgs[src_trg[1]].append(srcAndTrg[1].strip('\n'))
        data_dict[name]=Dataset.from_dict(srctrgs)
    return DatasetDict(data_dict)