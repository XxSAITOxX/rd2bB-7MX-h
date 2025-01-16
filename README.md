# SuperResolution-ResNet
## 概要
PyTorchを用いたResNetを実装。
目的は、物体検出で事前学習された重みは超解像に用いることが可能か検証。
可能な限り、今後も使えるフォーマットとして実装に取り組む。
## アップデート
- 2024.12.04 ページ作成
- 2025.01.17 学習と推論が可能
## 要件
- Python3.10.1
- Docker
- cuda 11.8以降
- DockerImage:nvidia/cuda:11.8.0-base-ubuntu22.04
- GPU：NVIDIA 2080 Super
## ファイルの説明
## 参考文献
- resnet.py
  - [ここのresnet18.py](https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch/tree/main)
    -  Kataoka Hirokatsu, et al."Pre-training without Natural Images" In ACCV2020.
    -  Kataoka Hirokatsu, et al."Pre-training without Natural Images" In IJCV2022.
- resnet18.py
  - resnet.py
  - [Pytorch Template 個人的ベストプラクティス](https://qiita.com/takubb/items/7d45ae701390912c7629)
- create_weight.py
  - [重みの書き換え](https://qiita.com/mathlive/items/d9f31f8538e20a102e14)
  - [nn.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)
