import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelRF.datasets import neural_rgbd_dataset

a = neural_rgbd_dataset('data','breakfast_room')
print(a.get_hwf())
print(a.get_HWK())