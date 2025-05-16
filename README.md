# 主要的檔案
- config.py
- train.py
- util.py
- vae.py
- LSTM PPO
    - actor.py
    - learner.py
- VRNN PPO
    - actor_vrnn.py
    - learner_vrnn.py

# 安裝說明

使用requirement.txt安裝套件
```
# python version == 3.10.15
pip install -r requirements.txt
```
>這裡使用的是cuda的pytorch，但程式碼內還是使用CPU

# 訓練

```
python train.py # lstm ppo
python train_vrnn.py # vrnn ppo
```

> 訓練好的log會儲存在"logs"資料夾中，模型和錄影會記錄在"models"資料夾中，
重新訓練同一個模型時會出現提示訊息，確定的話輸入y並按下ENTER就可以繼續

使用tensorboard來查看log

```
tensorboard --logdir logs
```

超參數以及檔案路徑等等的調整主要是在config.py中
