#ベースとなるイメージ
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

#必要そうなものをinstall
RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential libreadline-dev \ 
libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev

#任意のバージョンのpython
RUN wget --no-check-certificate https://www.python.org/ftp/python/3.10.1/Python-3.10.1.tgz \
&& tar -xf Python-3.10.1.tgz \
&& cd Python-3.10.1 \
&& ./configure --enable-optimizations\
&& make \
&& make install

#ライブラリのインポート
RUN python -m pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


#余分なパッケージの削除
RUN apt-get autoremove -y