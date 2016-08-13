

from pylab import *
data = random((3,3))
figure(1)
imshow(data, interpolation='none')
figure(2)
pcolor(flipud(data))
