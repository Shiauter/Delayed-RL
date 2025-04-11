# 主要的檔案
- actor_vrnn.py
- actor.py
- config.py
- learner_vrnn.py
- learner.py
- train.py
- util.py
- vae.py

# 安裝說明

使用requirement.txt安裝套件
```
# python version == 3.10.15
pip install -r requirements.txt
```
>這裡使用的是cuda的pytorch，但程式碼內還是使用CPU

# 訓練

```
python train.py
```

> 訓練好的log會儲存在"logs"資料夾中，模型和錄影會記錄在"models"資料夾中，  
但因為在讀取訓練好的model時有點問題還沒修，所以目前還只能從頭開始訓練，  
重新訓練同一個模型時會出現提示訊息，確認是否"刪除"先前的log、model和record，  
確定的話，輸入y並按下ENTER就可以繼續  

使用tensorboard來查看log

```
tensorboard --logdir logs
```

超參數的調整主要是在config.py中
