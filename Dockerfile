FROM bvlc/caffe:cpu

COPY ./requirements.txt /root/app/requirements.txt
WORKDIR /root/app

RUN pip install -r requirements.txt

COPY . /root/app

