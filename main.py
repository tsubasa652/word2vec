from fastapi import FastAPI
from typing import List, Tuple
from pydantic import BaseModel
from gensim.models import KeyedVectors
import uvicorn

model = KeyedVectors.load_word2vec_format("jawiki.100.bin", binary=True)

class Item(BaseModel):
    positive: List[str] = []
    negative: List[str] = []
    topn: int = 1

app = FastAPI()

@app.post("/")
async def root(item: Item):
    item = item.dict()
    try:
        res = model.most_similar(positive=item["positive"], negative=item["negative"], topn=item["topn"])
        res = {
            "status": "Success",
            "result": res
        }
        return res
    except KeyError:
        return {"status": "Error", "msg": "入力されたワードはデータベースに存在しませんでした。"}
    except AttributeError as e:
        return {"status": "Error", "msg": "エラーが発生しました。"}
    except ValueError as e:
        return {"status": "Error", "msg": "ワードが入力されていません。"}
    except:
        return {"status": "Error", "msg": "Unknown Error"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
