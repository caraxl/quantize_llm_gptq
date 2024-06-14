from typing import List,Dict
import json

def load_json(dataPath: str) -> List[Dict]:
    '''
    功能：加载jsonl格式的文件，返回list of dicts
    :param dataPath:
    :return:
    '''
    with open(dataPath, encoding="utf-8") as f:
        data = json.load(f)
    return data