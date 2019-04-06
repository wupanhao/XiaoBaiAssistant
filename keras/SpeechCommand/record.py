#!coding:utf-8
import os
import time

def recording(lists):
	username = input("Your Name : ")
	for item in lists:
		data_dir = './data_raw/'
		ok = 'n'
		while ok!='y':
			wait = input("Press Enter to start recording for " + item)
			dtime = time.strftime('%y_%m_%d_%H_%M_%S',time.localtime())			
			print('Please Speaking~')
			file_name = data_dir + item + '_' + username + '_' + dtime + '.wav'
			os.system('arecord -d 1 -r 16000 -c 1 -t wav -f S16_LE ' + file_name)
			print('Recording end , play the recording')
			os.system('aplay '+file_name)
			ok = input("is that OK?(y/n):")
			if ok!=y : 
				os.system('rm ' + file_name)
				print("bad file deleted")
			else :
				print('saved as ' + file_name)

if __name__ == '__main__':
	commands = ['打开','关闭','开灯','关灯','温度','湿度','门窗','音乐','笑话','天气']
	recording(commands)
