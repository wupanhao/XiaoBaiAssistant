# XiaoBaiAssistant(一个基于百度语音API的简单智能音箱实现方案)
- 核心代码不足100行，刚入门Python也能看懂
- 物联网+人工智能学习参考项目

## 基本设置
如果是使用同款WM8960音频扩展板(带小喇叭)
```
运行如下命令安装音频扩展板的驱动
git clone https://github.com/waveshare/WM8960-Audio-HAT
cd WM8960-Audio-HAT
sudo ./install.sh 
sudo reboot
```
也可以通过USB声卡或USB摄像头接入麦克风
然后你需要编辑自己HOME目录下的.asoundrc文件指定默认播放录音的设备，大概是这个样子的
音频输入输出接口可以通过`arecord -l` 和 `aplay -l`查看
```
pcm.!default {
  type asym
  capture.pcm "mic"
  playback.pcm "speaker"
}
pcm.mic {
  type plug
  slave {
    pcm "hw:1,0"
  }
}
pcm.speaker {
  type plug
  slave {
    pcm "hw:1,0"

  }
}
```
运行如下命令克隆代码
```
git clone https://github.com/wupanhao/XiaoBaiAssistant
cd XiaoBaiAssistant/
cp _snowboydetect_py3.so _snowboydetect.so
cp config.yaml.example config.yaml
```
前往百度语音开放平台、图灵机器人官网注册账号，在config.yaml填写自己的API密钥等信息
## 环境搭建(基于Docker)
```
运行如下命令安装Docker(带keras环境)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo docker run --rm -it --name="xiaobai" --privileged --net=host -v /home/pi/XiaoBaiAssistant:/xiaobai wupanhao/xiaobai-assistant:v1 env LANG=C.UTF-8 /usr/local/bin/jupyter-notebook --ip 0.0.0.0 --allow-root --notebook-dir /xiaobai/notebook/ 
之后打开http://[树莓派ip]:8888进入Jupyter Notebook的环境
```
## 环境搭建(原生系统,不带keras环境)
```
sudo apt install python3-pyaudio libatlas-base-dev libglib2.0-dev
sudo pip3 install pyyaml baidu-aip broadlink bluepy
cd ~/XiaoBaiAssistant
python3 demo.py
```
