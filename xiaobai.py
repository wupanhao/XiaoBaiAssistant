import sys
import os
import yaml
import random
from aip import AipSpeech
root_dir = "/xiaobai/"
sys.path.append(root_dir)
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
            notify_sound = root_dir+'resources/greetings/'+self.greetings[n]
            os.system("mpg123 "+notify_sound)   
            if self.callback is None:
                res = self.listen_and_recognize()
                if res == "":
                    self.speak("小白没听清呢")
                else:
                    print("你："+res)
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
            print('Listening...')
            self.detector.start(detected_callback=self._callback,sleep_time=0.03)
        except KeyboardInterrupt:
            print('stop')
        finally:
            self.detector.terminate()            
    #录音和识别函数,调用arecord录音
    def listen_and_recognize(self,length = 4):
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