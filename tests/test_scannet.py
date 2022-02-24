import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelRF.datasets import Room_SCANNET

a = Room_SCANNET('demo_dir','scannet')