import re
import yaml
import sys
import requests
import json
import time
root_dir = '/home/pi/workspace/XiaoBaiAssistant/'
sys.path.append(root_dir)
from demo import BaseSkill,XiaoBai,MusicSkill,TalkSkill
class HassSkill(BaseSkill): 
    def __init__(self):
        with open('./config.yaml') as f:
          config = yaml.load(f)['hass']
          self.url = config['url']
          self.port = config['port']
          self.token = config['api_token']
    #继承BaseSkill类，必须定义has_intent和action方法
    def has_intent(self,text=""):
        keywords = ["打开","关闭","查询"]
        for i in keywords:
            if text.find(i)>=0:
                print("handled by hass")
                return True
        return False
    def action(self,text="",callback=print):
        #headers = {'x-ha-access': password, 'content-type': 'application/json'}
        headers = {'Authorization': "Bearer "+self.token, 'Content-Type': 'application/json'}
        p = json.dumps({"text": text})
        r = requests.post(self.url + ":" + self.port + "/api/services/conversation/process", headers=headers,data=p)
        print(r.text)
if __name__ == '__main__':
  s = HassSkill()
  s.handle("打开卫生间的灯")
  time.sleep(2)
  s.handle("关闭卫生间的灯")
  #keyword_model = root_dir+'resources/models/snowboy.umdl'
  #xiaobai = XiaoBai(keyword_model=keyword_model)
  #xiaobai.add_skill(MusicSkill())
  #xiaobai.add_skill(HassSkill())
  #xiaobai.listen_for_keyword()
