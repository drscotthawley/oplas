
# Numpy original from https://github.com/tulip-control/polytope/blob/main/polytope/polytope.py
# Converted to PyTorch by SHH

import torch
import math
import torch.nn as nn


def givens_rotation_matrix(i, j, theta, N):
    """Return the Givens rotation matrix for an N-dimensional space."""
    R = torch.eye(N)
    c = torch.cos(theta)
    s = torch.sin(theta)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R

def solve_rotation_ap(u, v, check_vecs=False, debug=True):
    """Return the rotation matrix for the rotation in the plane defined by the
    vectors u and v across TWICE the angle between u and v.

    This algorithm uses the Aguilera-Perez Algorithm \cite{Aguilera} (https://dspace5.zcu.cz/bitstream/11025/6178/1/N29.pdf)
    to generate the rotation matrix. The algorithm works basically as follows:

    Starting with the Nth component of u, rotate u towards the (N-1)th
    component until the Nth component is zero. Continue until u is parallel to
    the 0th basis vector. Next do the same with v until it only has none zero
    components in the first two dimensions. The result will be something like
    this:

    [[u0,  0, 0 ... 0],
     [v0, v1, 0 ... 0]]

    Now it is trivial to align u with v. Apply the inverse rotations to return
    to the original orientation.

    NOTE: The precision of this method is limited by sin, cos, and arctan
    functions.
    Also NOTE: Reversing order of u,v -> v,u yields R.T
    """
    u, v = u.squeeze(), v.squeeze()
    assert len(u.shape)==1, f"u ({list(u.shape)}) & v ({list(v.shape)}) should be single vectors, not batches"

    # BTW: pretty safe to assume u & v have same dims, on same device
    device = u.device
    N = len(u)                       # the number of dimensions
    M = torch.eye(N).to(device)      # stores rotation matrix

    # optional: maybe save a bit of time for (anti-)parallel or zero u & v
    if check_vecs and u.norm() * v.norm() == torch.dot(u,v).abs(): 
        if debug:
            print(f"solve_rotation_ap: zero or (anti-)parallel u,v: 0 degree rotation")
        return M 

    uv = torch.stack([u, v], axis=1)  # the plane of rotation
    # ensure u has positive basis0 component
    if uv[0, 0] < 0:
        M[0, 0] = -1
        M[1, 1] = -1
        uv = M.matmul(uv)
    # align uv plane with the basis01 plane and u with basis0.
    for c in range(2):
        for r in range(N - 1, c, -1):
            if uv[r, c] != 0:  # skip rotations when theta will be zero
                theta = torch.arctan2(uv[r, c], uv[r - 1, c])
                Mk = givens_rotation_matrix(r, r - 1, theta, N).to(device)
                uv = Mk.matmul(uv)
                M = Mk.matmul(M)
    # rotate u onto v
    theta = 2 * torch.arctan2(uv[1, 1], uv[0, 1])
    if debug:
        print(f"solve_rotation_ap: {180 * theta / math.pi:6.2f} degree rotation")
    R = givens_rotation_matrix(0, 1, theta, N).to(device)
    # perform M rotations in reverse order
    M_inverse = M.T
    R = M_inverse.matmul(R.matmul(M))
    return R


def rotate_batch(R, v_batch):
    #return v_batch @ R  # we could multiply from the right (aka "passive rotations")
    #^But conventionally, ("active") rotations are applied by multiplying from the left, so...
    #return (R @ v_batch.T).T      #  ...but (A @ B).T = B.T @ A.T, so then...
    return v_batch @ R.T   # linear algebra FTW





def get_rot_2d(theta, nd=2):
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.tensor([ [c, -s],[s,c] ])

def get_rot_nd(u, v, debug=False):
    """Return the rotation matrix that rotates u onto v."""
    return solve_rotation_ap(u, v, debug=debug)

class FiLMR2d(nn.Module):
    "affine transformation plus rotation, in 2d"
    def __init__(self, nd=2, 
                 beta_init_fac = 0.0001, # tiny beta for FILM is maybe cheating
                 theta_init_fac=6.28/15, # 2pi/n init is "cheating" a bit
                 ):
        super().__init__()
        self.gamma =  nn.Parameter(torch.ones((1)))
        self.beta =  beta_init_fac*nn.Parameter(torch.randn((1)))
        self.theta = theta_init_fac*nn.Parameter( torch.ones((1)) ) 

    def forward(self, x):
        rot = get_rot(self.theta).to(x.device)
        return (x * self.gamma + self.beta.to(x.device)) @ rot


class FiLMRnd(nn.Module):
    "affine transformation plus rotation, in nd"
    def __init__(self, nd=3, 
                 beta_init_fac = 0.0001, # tiny beta is maybe cheating
                 uv_diff_fac = 0.25, # difference scale between initial u and v
                 norm_uv = False,
                 use_bivector=True,
                 rank_uv=1, # just an option I tried. leave it at 1
                 ):
        super().__init__()
        self.norm_uv, self.use_bivector = norm_uv, use_bivector
        self.gamma =  nn.Parameter(torch.ones((1)))
        self.beta =  beta_init_fac*nn.Parameter(torch.randn((1)))
        #self.theta = theta_init_fac*nn.Parameter( torch.ones((1)) ) 
        
        self.u = torch.randn((rank_uv,nd)) 
        self.v = self.u + uv_diff_fac*torch.randn((rank_uv,nd))   # make v a bit different from u
        if self.norm_uv:
            # FYI: normalizing u & v, either initially or at all times, doesn't magically fix everything but we can try it
            self.u = self.u / self.u.norm(dim=-1)
            self.v = self.v / self.v.norm(dim=-1)
        self.u = nn.Parameter( self.u )
        self.v = nn.Parameter( self.v ) 

    def forward(self, x, debug=False):
        u,v = self.u, self.v
        if self.use_bivector:
            if len(u.shape)<2:
                u, v = u.unsqueeze(0), v.unsqueeze(0)
            rot = 0.5 * ( u.T @ v - v.T @ u)
        else:
            #rot = get_rot(self.theta).to(x.device)
            rot = get_rot_nd(self.u, self.v, debug=debug).to(x.device)
        if debug: print("self.u.shape, self.v.shape, rot.shape =",self.u.shape, self.v.shape,rot.shape)
        return (x * self.gamma + self.beta.to(x.device)) @ rot