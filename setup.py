import subprocess
import os

def install(name) :
    subprocess.call(['pip','install',name])

print('Starting to install pip modules...')
os.system('python get-pip.py')

print('Starting to install required python modules...')
file = open('requirements.txt','r')

for line in file :
    install(line)
print('Installation completed.')
file.close()
