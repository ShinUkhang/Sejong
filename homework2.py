import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import os
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

files = os.listdir('./')

student=['선생님', '김혜영', '박지윤', '송주아', '오세은', '이다은','이인원', '이혜연', '정나윤','성치영']

i=0
sr=16000
for file in files:
    if 'student' in file:
        print(file)
        i += 1
        y, sr = librosa.load(file, sr=sr)
        globals()['mfcc{}'.format(i)]= librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24, hop_length=512)
        mfccdata=globals()['mfcc{}'.format(i)]
        globals()['gmm{}'.format(i)] = GaussianMixture(n_components=5, covariance_type='diag', tol=0.001, max_iter=200,
                               init_params='random').fit(mfccdata.T)



testfile='test0.wav'
yt, srt=librosa.load(testfile, sr=8000)
mfcctest=librosa.feature.mfcc(y=yt, sr=srt, n_mfcc=24, hop_length=512)

p=-100000000
pval=[]
for j in range(1,i+1):
    gmmodel=globals()['gmm{}'.format(j)]
    pt=gmmodel.score(mfcctest.T)
    pval.append(pt)
    print(pt)
    if pt > p:
        p=pt
        n=j-1


#결과를 위한 시각화
print(student[n]+'의 목소리일 확률은 {}으로 가장 높습니다.'.format(round(p,2)))
plt.figure(dpi=100)
plt.subplot(3,1,1)
plt.title(student[n]+'의 목소리일 확률은 {}으로 가장 높습니다.'.format(round(p,2)))
librosa.display.specshow(globals()['mfcc{}'.format(n+1)])
plt.subplot(3,1,2)
librosa.display.specshow(mfcctest)
plt.subplot(3,1,3)
plt.plot(student,pval, 'r--o')
plt.grid()
plt.show()