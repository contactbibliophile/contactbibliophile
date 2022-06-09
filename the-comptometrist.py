import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from decimal import Decimal
import matplotlib.pyplot as plt
import math

def find_ecc(image): #function to find elongation
  img = cv2.imread(image)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret,thresh = cv2.threshold(img,1,255,cv2.THRESH_BINARY_INV)
  contours,hierarchy = cv2.findContours(thresh, 1, 2)
  cnt = contours[0]
  k = cv2.isContourConvex(cnt)
  ellipse = cv2.fitEllipse(cnt)
  im = cv2.ellipse(img,ellipse,(0,255,0),2)
  (x,y),(b,a),angle = cv2.fitEllipse(cnt)
  ecc = max(a,b)/min(a,b)
  return ecc

def find_elongation(image): #function to find elongation
  img = cv2.imread(image)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (7, 7), 0)
  ret,thresh = cv2.threshold(blurred,254,255,cv2.THRESH_BINARY)
  from google.colab.patches import cv2_imshow
  contours,hierarchy = cv2.findContours(thresh, 1, 2)
  cnt = contours[0]
  k = cv2.isContourConvex(cnt)
  ellipse = cv2.fitEllipse(cnt)
  im = cv2.ellipse(img,ellipse,(0,255,0),2)
  (x,y),(ma,Ma),angle = cv2.fitEllipse(cnt)
  return Ma/ma

estElo = []
for i in range(20):
  fileName = "p"+str((i+1))+"R.PNG"
  estElo.append(find_ecc(fileName))
for i in range(30):
  if i == 4 or i ==5 or i==7 or i==11:
    estElo.append(1.41)
  else:
    append = find_elongation("Myopia_marked ("+str(i+1)+").jpg")
    estElo.append(append)
#regression training
from sklearn.metrics import r2_score
x=estElo
y = [5,1,0.25,0.5,5.25,0.25,4,3.75,2.5,0,0,1.25,1,1.25,2.25,0,2,0.25,7,0,4.5,3.75,8,5.25,3.75,3.75,4,3.75,3.1,2.8,1.3,3.75,10.5,8.6,7.9,7.8,5,5,6.3,6.3,1,1,2.5,2.5,0.8,1.8,3.9,3.8,17,15]
print(len(x),len(y))
mymodel = np.poly1d(np.polyfit(x, y, 2))
myline = np.linspace(1, 5, 100)
plt.scatter(x, y)
plt.plot(myline,mymodel(myline))
plt.show()
plt.plot(mymodel(x))
avg =0
for i in range(len(x)):
  err = y[i]-mymodel(x[i])
  if err >= 0:
    print(i+1,mymodel(x[i]),y[i],err)
  elif err < 0:
    print(i+1,mymodel(x[i]),y[i],-err)
  avg+=math.fabs(err)
print('R^2: '+str(r2_score(y, mymodel(x))))
print('Mean Error: '+str(avg/39))
#create list of estimated (calculated by program) elongations for testing
estElo = []
for i in range(20):
  fileName = str('p'+str(i+1)+'L.PNG')
  est = find_ecc(fileName) 
#  print(fileName,est)
  estElo.append(est)
#https://www.aaopt.org/detail/knowledge-base-article/myelinated-nerve-retinal-fibers-associated-myopia-astigmatism-and-anisometropic-amblyopia justification 
#regression and error calculation
error = 0
rights = [7.25,0.75,0,0.50,5.50,0.50,4,2.50,2.50,0.50,0,1.75,1.25,1.00,2.25,
0.50,1.00,0.25,7.00,0.25]
powers = []
mad = 0
for i in range(len(estElo)): 
  estPow = mymodel(estElo[i])
  powers.append(estPow)
  realPow = Decimal(rights[i])
  err = Decimal(math.fabs(Decimal(realPow)-Decimal(estPow)))
  mad += err

plt.scatter(estElo,rights)
plt.scatter(estElo,powers)
print("Average Error in Testing: "+str(mad/20))
