import re
import sys
root_dir = '/home/pi/workspace/XiaoBaiAssistant/'
sys.path.append(root_dir)
from demo import BaseSkill,XiaoBai,MusicSkill,TalkSkill
from LEGO_Car import Car
class CarSkill(BaseSkill): 
    def __init__(self):
        self.car = Car()
        self.car.set_speed(50)
    #继承BaseSkill类，必须定义has_intent和action方法
    def has_intent(self,text=""):
        keywords = ["前进","后退","左转","右转"] #如果说了 我想听、播放 一类的词就认为有播放音乐的意图，再由action函数判断应该播放哪一首歌曲
        for i in keywords:
            if text.find(i)>=0:
                print("handled by car")
                return True
        return False
    def action(self,text="",callback=print):
        m = re.search('(前进|后退|左转|右转)(\d+)(cm$|度$)', text)
        if m is not None:
            action = m.groups()[0]
            param = int(m.groups()[1])
            if action == "前进":
                self.car.go_forward(param)
            elif action == "后退":
                self.car.go_backward(param)
            elif action == "左转":
                self.car.turn_left(param)
            elif action == "右转":
                self.car.turn_right(param)
if __name__ == '__main__':
  keyword_model = root_dir+'resources/models/snowboy.umdl'
  xiaobai = XiaoBai(keyword_model=keyword_model)
  xiaobai.add_skill(MusicSkill())
  xiaobai.add_skill(CarSkill())
  #xiaobai.add_skill(TalkSkill())
  xiaobai.listen_for_keyword()
