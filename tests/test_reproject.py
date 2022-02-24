import torch

def inv_project_points_cam_coords(intrinsic, uvd, c2w):
    '''
    As inv_project_points but doesn't do the homogeneous transformation
    int: 3x3, uvd:Nx3, c2w:3X4
    '''
    if len(uvd.shape) > 2:
        uvd = uvd.reshape(-1,3)
    if intrinsic.shape[0]>3:
        intrinsic = intrinsic[:3,:3]
    if c2w.shape[0]>3:
        c2w = c2w[:3,:]
    n_points = uvd.shape[0]

    # creating the camera rays
    uv1 = torch.hstack((uvd[:, :2], torch.ones((n_points, 1))))
    camera_rays = torch.mm(intrinsic.inverse(), uv1.T).T

    # forming the xyz points in the camera coordinates
    temp = uvd[:, 2][:,None].repeat(1,3)
    xyz_at_cam_loc = temp * camera_rays
    xyz_at_cam_loc = torch.hstack((xyz_at_cam_loc, torch.ones((n_points, 1))))
    # project to xyz
    xyz_world = torch.mm(c2w, xyz_at_cam_loc.T).T

    return xyz_world #Nx3

def projection(xyz, K, ext):
    newK = torch.eye(4)
    newK[:3,:3] = K
    P = newK@ext
    uvz = P@xyz
    return uvz, uvz[:2,:]/uvz[2:3,:]

K = torch.tensor([[300,0,400],[0,300,400],[0,0,1]]).float()
R = torch.eye(3)
pose = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,-1],[0,0,0,1]]).float()
print(pose)

Pw = torch.tensor([[3,4,4,1]]).float().T
print(Pw)

a,b = projection(Pw,K,pose.inverse())

place = torch.ones((3,1))
place[:2,] = b
wd = inv_project_points_cam_coords(K,torch.tensor([580,640,5]).unsqueeze(0),pose)
print(wd.shape)

assert torch.abs(Pw[:3]-wd.T).sum()<0.0001



