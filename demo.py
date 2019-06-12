import sys
import os
import yaml
import random
from aip import AipSpeech
root_dir = './'
import snowboydecoder
class XiaoBai:
    #初始化函数，设置关键字模型，
    def __init__(self,keyword_model,callback=None):
        self.detector = snowboydecoder.HotwordDetector(keyword_model, sensitivity=0.5)
        self.skills = []
        self.greetings = ["嗯哼.mp3","我在.mp3","请说.mp3"]
        self.callback = callback
        with open(root_dir+"config.yaml") as f:
            config = yaml.load(f)['baidu_yuyin']
            self.client = AipSpeech(config['app_id'], config['api_key'], config['secret_key'])
    #检测到关键字后的操作
    def _callback(self):
            self.detector.terminate()
            n = random.randint(0,len(self.greetings)-1)
            #notify_sound = root_dir+'resources/greetings/'+self.greetings[n]
            #os.system("mpg123 "+notify_sound)
            notify_sound = root_dir+'resources/ding.wav'
            os.system("aplay "+notify_sound)
            if self.callback is None:
                res = self.listen_and_recognize()
                if res == "":
                    self.speak("小白没听清呢")
                else:
                    print(res)
                    handled = False
                    for skill in self.skills:
                        if skill.handle(res,callback=self.speak):
                            handled = True
                            break
                    if not handled:
                        self.speak("小白暂时不会处理呢")
            else:
                self.callback()
            self.detector.start(detected_callback=self._callback,sleep_time=0.03)
    #添加技能
    def add_skill(self,skill):
        if skill.type == "skill":
            self.skills.append(skill)
    def listen_for_keyword(self):
        try:
            print('等待唤醒...')
            self.detector.start(detected_callback=self._callback,sleep_time=0.03)
        except KeyboardInterrupt:
            print('stop')
        finally:
            self.detector.terminate()            
    #录音和识别函数,调用arecord录音
    def listen_and_recognize(self,length = 3):
        print('你：',end="")
        os.system("arecord -d %d -r 16000 -c 1 -t wav -f S16_LE record.wav" % (length,) )    
        with open("./record.wav", 'rb') as fp:
            res = self.client.asr(fp.read(), 'wav', 16000, { 'dev_pid': 1536,})
            if isinstance(res, dict) and res['err_no']==0:
                return res["result"][0]
            else:
                #print(res)
                return ""
    #调用百度语音合成API进行回复
    def speak(self,text = '你好呀',lang = 'zh',type = 1 , vol = 5, spd = 5 , pit = 5):
        result  = self.client.synthesis(text, lang, type, {'vol': vol,'spd':spd,'pit':pit})
        # 识别正确返回语音二进制 错误则返回dict
        if not isinstance(result, dict):
            with open('speak.mp3', 'wb') as f:
                f.write(result)
            print('小白：'+text)
            os.system('mpg123 speak.mp3')
        else:
            print('emmmm，小白出错了呢',result)
            
import abc #利用abc模块实现抽象类
#编写扩展技能的基本格式，has_intent函数检测是否有需要该技能处理的意图，action函数执行对应的处理
class BaseSkill(metaclass=abc.ABCMeta):
    type='skill'
    #参数说明 
    #    text：语音识别的到的文本
    #    callback：反馈文本的处理函数，默认直接打印，也可以传入语音合成函数进行语音回复
    #定义抽象方法，检测是否有需要该技能处理的意图
    @abc.abstractmethod 
    def has_intent(self,text=""):
        pass
    #定义抽象方法，根据意图处理处理信息
    @abc.abstractmethod
    def action(self,text=""):
        pass
    #处理函数，根据意图处理处理信息，返回是否继续检测意图
    def handle(self,text="",callback=print):
        if self.has_intent(text=text):
            self.action(text=text,callback=callback)
            return True
        else:
            return False
import requests
import json
import yaml

class TalkSkill(BaseSkill):
    def __init__(self):
        #root_dir = "/xiaobai/"
        with open(root_dir+"config.yaml") as f:
            config = yaml.load(f)['tuling']
            self.key = config['key']
            self.url = 'http://www.tuling123.com/openapi/api'
    #继承BaseSkill类，必须定义has_intent和action方法
    def has_intent(self,text=""):
        if text!= "":
            return True
        return False
    def action(self,text="",callback=print):
        try:
            req = {'key':self.key,'info':text}
            res = requests.post(url = self.url, data = req)
            #print(res.text)
            jd = json.loads(res.text)
            callback(jd['text'])
        except:
            callback("出错了呢，可能网络不太好")
import os
import argparse
import random
import subprocess
import re
import sys

class MusicSkill(BaseSkill): 
    #递归找出指定文件夹下的所有mp3文件
    def getfiles(self,dir): 
        lists = []
        for item in os.listdir(dir): 
            path = os.path.join(dir, item) 
            if os.path.isdir(path): 
                lists.extend(self.getfiles(path))
            elif os.path.splitext(path)[1]==".mp3" :
                lists.append(path)
        return lists
    #根据关键字查找对应的mp3文件，如果有多个匹配项，随机返回一个
    def find_music(self,keyword = ""):
        result=[]
        for item in self.lists:
            if item.find(keyword) >= 0:
                result.append(item)
        if len(result) > 0:
            n = random.randint(0,len(result)-1)
            music = result[n]
            return music
        else:
            return None
    #调用系统命令播放音乐
    @staticmethod
    def play_music(path):
        os.system('mpg123 "'+ path+'"')    
    def __init__(self):
        #path = "/xiaobai/resources/music"
        path = root_dir+"resources/music"
        self.lists = self.getfiles(path)
    #继承BaseSkill类，必须定义has_intent和action方法
    def has_intent(self,text=""):
        keywords = ["我想听","播放"] #如果说了 我想听、播放 一类的词就认为有播放音乐的意图，再由action函数判断应该播放哪一首歌曲
        for i in keywords:
            if text.find(i)>=0:
                return True
        return False
    def action(self,text="",callback=print):
        m = re.search('(我想听|播放)(.+?)(的歌$|$)', text)
        if m is not None:
            search = m.groups()[1]
            result = self.find_music(search)
            if result is not None:
                callback("找到，"+os.path.basename(result).replace('.mp3','')+",为您播放")
                self.play_music(result)
            else:
                callback("未找到",search)
                
import broadlink,binascii
class SwitchSkill(BaseSkill):
    def __init__(self):
        #print(mac_addr)
        devices = broadlink.discover(timeout=5)
        if len(devices)>0:
            print("find dev")
            self.sw = broadlink.sp2(devices[0].host,devices[0].mac,None)
        else:
            print("auto discover faind , using default switch setting")
            mac_addr = binascii.unhexlify("78:0f:77:c8:b4:c0".encode().replace(b':', b''))
            self.sw = broadlink.sp2(('192.168.43.114',80),mac_addr,None)
        if self.sw.auth()!=True:
            print('认证失败，请重试')
    #继承BaseSkill类，必须定义has_intent和action方法
    def has_intent(self,text=""):
        for keyword in ["打开","关闭","开灯","关灯"]:
            if text.find(keyword)>= 0:
                return True
        return False
    def action(self,text="",callback=print):
        try:
            for keyword in ["打开","开灯"]:
                if text.find(keyword)>= 0:
                    self.sw.set_power(1)
                    callback("已执行，"+text)
                    return
            for keyword in ["关闭","关灯"]:
                if text.find(keyword)>= 0:
                    self.sw.set_power(0)      
                    callback("已执行，"+text)
                    return
        except:
            callback("出错了呢，可能网络不太好")

from bluepy import btle
import re

class XiomiHygroThermoDelegate(object):
    def __init__(self):
        self.temperature = None
        self.humidity = None
        self.received = False

    def handleNotification(self, cHandle, data):
        #print(data)
        if cHandle == 14:
            m = re.search('T=([\d\.]*)\s+?H=([\d\.]*)', ''.join(map(chr, data)))
            self.temperature = m.group(1)
            self.humidity = m.group(2)
            self.received = True
            
class SensorSkill(BaseSkill):
    def __init__(self):
        self.address = "58:2d:34:30:36:d1"
    #继承BaseSkill类，必须定义has_intent和action方法
    def has_intent(self,text=""):
        for keyword in ["温度","湿度","温湿度","气温"]:
            if text.find(keyword)>= 0:
                return True
        return False
    def action(self,text="",callback=print):
        try:
            p = btle.Peripheral(self.address)
            delegate = XiomiHygroThermoDelegate()
            p.withDelegate( delegate )
            p.writeCharacteristic(0x10, bytearray([1, 0]), True)
            while not delegate.received:
                p.waitForNotifications(30.0)
            temperature = delegate.temperature
            humidity = delegate.humidity
            callback("温度："+temperature+"℃，湿度："+humidity+"%")
        except:
            callback("出错了呢")
if __name__ == '__main__':
  keyword_model = root_dir+'resources/models/snowboy.umdl'
  xiaobai = XiaoBai(keyword_model=keyword_model)
  xiaobai.add_skill(MusicSkill())
  #xiaobai.add_skill(SwitchSkill())
  #xiaobai.add_skill(SensorSkill())
  xiaobai.add_skill(TalkSkill())
  xiaobai.listen_for_keyword()
