from raspbian/stretch
#RUN apt update 
RUN apt update &&  apt install alsa-utils python3 python3-pyaudio python3-pip libatlas-base-dev libglib2.0-dev mpg123 -y 
RUN pip3 install pyyaml baidu-aip
#	pip3 install  pyyaml baidu-aip broadlink bluepy
#	echo  "export LANG=C.UTF-8" >> .profile 
COPY ./ /XiaobaiAssistant
CMD bash -c "cd /XiaobaiAssistant  && LANG=C.UTF-8 /usr/bin/python3 xiaobai.py" 
#ENTRYPOINT bash -c "cd /XiaobaiAssistant  && LANG=C.UTF-8 /usr/bin/python3 demo.py" 
# -v ~/.asoundrc:/root/.asoundrc
#test with docker run -it  --rm -v ~/.asoundrc:/root/.asoundrc --privileged wupanhao/xiaobai:v0.1
