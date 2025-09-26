import torch

from typing import Union
from wis3d import Wis3D
from pytorch3d import transforms


class HWis3D(Wis3D):
    """ Abstraction of Wis3D for human motion. """

    def __init__(
        self,
        out_path    : str   = 'wis3d',
        seq_name    : str   = 'debug',
        xyz_pattern : tuple = ('x', 'y', 'z'),
    ):
        seq_name = seq_name.replace('/', '-')
        super().__init__(out_path, seq_name, xyz_pattern)

    def add_motion_verts(
        self,
        verts : torch.Tensor,
        name  : str,
        offset: int = 0,
    ):
        """
        Add sequence of vertices to the wis3d viewer.

        ### Args
        - `verts`: torch.Tensor
            - shape = (L, V, 3), L is the sequence length, V is the number of vertices
        - `name`: str
            - the name of the point cloud
        - `offset`: int, default = 0
            - the offset for the sequence index
        """
        assert (len(verts.shape) == 3), 'The input `verts` should have 3 dimensions: (L, V, 3).'
        assert (verts.shape[-1] == 3), 'The last dimension of `verts` should be 3.'

        # Get the sequence length.
        L, V, _ = verts.shape
        verts = verts.detach().cpu()

        # Add vertices frame by frame.
        for i in range(L):
            self.set_scene_id(i + offset)
            self.add_point_cloud(
                vertices = verts[i],
                colors   = None,
                name     = name,
            )

        # Reset Wis3D scene id.
        self.set_scene_id(0)

    def add_motion_skel(
        self,
        joints : torch.Tensor,
        bones  : Union[list, torch.Tensor],
        colors : Union[list, torch.Tensor],
        name   : str,
        offset : int = 0,
    ):
        """
        Add sequence of joints with specific skeleton to the wis3d viewer.

        ### Args
        - `joints`: torch.Tensor
            - shape = (L, J, 3), L is the sequence length, J is the number of joints
        - `bones`: list
            - A list of bones of the skeleton, i.e. the edge in the kinematic trees. A bone is represented
              as
        - `colors`: list
        - `name`: str
            - the name of the point cloud
        - `offset`: int, default = 0
            - the offset for the sequence index
        """
        assert (len(joints.shape) == 3), 'The input `joints` should have 3 dimensions: (L, J, 3).'
        assert (joints.shape[-1] == 3), 'The last dimension of `joints` should be 3.'
        if isinstance(bones, list):
            bones = torch.tensor(bones)
        if isinstance(colors, list):
            colors = torch.tensor(colors)

        # Get the sequence length.
        L, J, _ = joints.shape
        joints = joints.detach().cpu() # (L, J, 3)

        # Add vertices frame by frame.
        for i in range(L):
            self.set_scene_id(i + offset)
            self.add_lines(
                start_points = joints[i][bones[:, 0]],
                end_points   = joints[i][bones[:, 1]],
                colors       = colors,
                name         = name,
            )

        # Reset Wis3D scene id.
        self.set_scene_id(0)

    def add_vec_seq(
        self,
        vecs    : torch.Tensor,
        name    : str,
        offset  : int = 0,
        seg_num : int = 16,
    ):
        """
        Add directional line sequence to the wis3d viewer.

        The line will be gradient colored, and the direction of the vector is visualized as from dark to light.

        ### Args
        - `vecs`: torch.Tensor
            - shape = (L, 2, 3), L is the sequence length, then give the start 3D point and end 3D point.
        - `name`: str
            - the name of the vector
        - `offset`: int, default = 0
            - the offset for the sequence index
        - `seg_num`: int, default = 16
            - the number of segments for gradient color, will just change the visualization effect
        """
        assert (len(vecs.shape) == 3), 'The input `vecs` should have 3 dimensions: (L, 2, 3).'
        assert (vecs.shape[-2:] == (2, 3)), 'The last two dimension of `vecs` should be (2, 3).'

        # Get the sequence length.
        L, _, _ = vecs.shape
        vecs = vecs.detach().cpu()

        # Cut the line into segments.
        steps_delta = (vecs[:, [1]] - vecs[:, [0]]) / (seg_num + 1) # (L, 1, 3)
        steps_cnt   = torch.arange(seg_num + 1).reshape((1, seg_num + 1, 1)) # (1, seg_num+1, 1)
        segs = torch.einsum('LxY,uSv->LSY', steps_delta, steps_cnt) # (L, seg_num+1, 3)
        segs = segs + vecs[:, [0]] # (L, seg_num+1, 3)
        start_pts = segs[:, :-1] # (L, seg_num, 3)
        end_pts   = segs[:, 1:] # (L, seg_num, 3)

        # Prepare the gradient colors.
        grad_colors = torch.linspace(0, 255, seg_num).reshape((seg_num, 1)).repeat(1, 3) # (seg_num, 3)

        # Add vertices frame by frame.
        for i in range(L):
            self.set_scene_id(i)
            self.add_lines(
                start_points = start_pts[i],
                end_points   = end_pts[i],
                colors       = grad_colors,
                name         = name,
            )

        # Reset Wis3D scene id.
        self.set_scene_id(0)

    def add_traj(
        self,
        positions : torch.Tensor,
        name      : str,
        offset    : int = 0,
    ):
        """
        Visualize the the positions change across the time as trajectory. The newer position will be brighter.

        ### Args
        - `positions`: torch.Tensor
            - shape = (L, 3), L is the sequence length
        - `name`: str
            - the name of the trajectory
        - `offset`: int, default = 0
            - the offset for the sequence index
        """
        assert (len(positions.shape) == 2), 'The input `positions` should have 2 dimensions: (L, 3).'
        assert (positions.shape[-1] == 3), 'The last dimension of `positions` should be 3.'

        # Get the sequence length.
        L, _ = positions.shape
        positions = positions.detach().cpu()
        traj = positions[[0]] # (1, 3)

        # Prepare the gradient colors.
        grad_colors = torch.linspace(208, 48, L).reshape((L, 1)).repeat(1, 3) # (L, 3)

        for i in range(L):
            traj = torch.cat((traj, positions[[i]]), dim=0) # (i+2, 3)
            self.set_scene_id(i + offset)
            self.add_lines(
                start_points = traj[:-1],
                end_points   = traj[1:],
                colors       = grad_colors[-(i+1):],
                name         = name,
            )

        # Reset Wis3D scene id.
        self.set_scene_id(0)

    def add_sphere_sensors(
        self,
        positions  : torch.Tensor,
        radius     : Union[torch.Tensor, float],
        activities : torch.Tensor,
        name       : str,
    ):
        """
        Draw the sphere sensors with different colors to represent the activities. The color is from white to red.

        ### Args
        - `positions`: torch.Tensor
            - shape = (N, 3), N is the number of sensors
        - `radius`: torch.Tensor or float
            - shape = (N,), N is the number of sensors
        - `activities`: torch.Tensor
            - shape = (N)
            - the activities of the sensors, from 0 to 1
        - `name`: str
            - the name of the spheres
        """
        assert (len(positions.shape) == 2), 'The input `positions` should have 2 dimensions: (N, 3).'
        assert (positions.shape[-1] == 3), 'The last dimension of `positions` should be 3.'
        N, _ = positions.shape
        if isinstance(radius, float):
            radius = torch.Tensor(radius).reshape(1).repeat(N) # (N)
        elif len(radius.shape) == 0:
            radius = radius.reshape(1).repeat(N)

        colors = torch.ones(size=(N, 3)).float()
        colors[:, 0] = 255
        colors[:, 1] = (1 - activities) ** 2 * 255
        colors[:, 2] = (1 - activities) ** 2 * 255
        self.add_spheres(
            centers = positions,
            radius  = radius,
            colors  = colors,
            name    = name,
        )

    def add_sphere_sensors_seq(
        self,
        positions  : torch.Tensor,
        radius     : Union[torch.Tensor, float],
        activities : torch.Tensor,
        name       : str,
        offset     : int = 0,
    ):
        """
        Draw the sphere sensors with different colors to represent the activities. The color is from white to red.

        ### Args
        - `positions`: torch.Tensor
            - shape = (L, N, 3), N is the number of sensors
        - `radius`: torch.Tensor or float
            - shape = (L, N,), N is the number of sensors
        - `activities`: torch.Tensor
            - shape = (L, N)
            - the activities of the sensors, from 0 to 1
        - `name`: str
            - the name of the spheres
        - `offset`: int, default = 0
            - the offset for the sequence index
        """
        assert (len(positions.shape) == 3), 'The input `positions` should have 3 dimensions: (L, N, 3).'
        assert (positions.shape[-1] == 3), 'The last dimension of `positions` should be 3.'
        L, N, _ = positions.shape

        for i in range(L):
            self.set_scene_id(i + offset)
            self.add_sphere_sensors(
                positions  = positions[i],
                radius     = radius,
                activities = activities[i],
                name       = name,
            )
    # ===== Overriding methods from original Wis3D. =====


    def add_lines(
        self,
        start_points: torch.Tensor,
        end_points  : torch.Tensor,
        colors      : Union[list, torch.Tensor] = None,
        name        : str   = None,
        thickness   : float = 0.01,
        resolution  : int   = 4,
    ):
        """
        Add lines by points. Overriding the original `add_lines` method to use mesh to provide browser from crash.

        ### Args
        - `start_points`: torch.Tensor
            - shape = (N, 3), N is the number of lines
        - `end_points`: torch.Tensor
            - shape = (N, 3), N is the number of lines
        - `colors`: list or torch.Tensor
            - shape = (N, 3)
            - the color of the lines, from 0 to 255
        - `name`: str
            - the name of the vector
        - `thickness`: float, default = 0.01
            - the thickness of the lines
        - `resolution`: int, default = 3
            - the 'line' was actually a poly-cylinder, and the resolution how it looks like a cylinder
        """
        if isinstance(colors, list):
            colors = torch.tensor(colors)

        assert (len(start_points.shape) == 2), 'The input `start_points` should have 2 dimensions: (N, 3).'
        assert (len(end_points.shape) == 2), 'The input `end_points` should have 2 dimensions: (N, 3).'
        assert (start_points.shape == end_points.shape), 'The input `start_points` and `end_points` should have the same shape.'

        # ===== Prepare the data. =====
        N, _ = start_points.shape
        device = start_points.device
        dir = end_points - start_points # (N, 3)
        dis = torch.norm(dir, dim=-1, keepdim=True) # (N, 1)
        dir = dir / dis # (N, 3)
        K = resolution + 1 # the first & the last point share the position
        # Find out directions that are negative to the y-axis.
        vec_y = torch.Tensor([[0, 1, 0]]).float().to(device) # (1, 3)
        neg_mask = (dir @ vec_y.transpose(-1, -2) < 0).squeeze() # (N,)

        # ===== Get the ending surface vertices of the cylinder. =====
        # 1. Get the surface vertices template in x-z plain.
        radius = torch.linspace(0, 2*torch.pi, K) # (K,)
        v_ending_temp = \
            torch.stack(
                [torch.cos(radius), torch.zeros_like(radius), torch.sin(radius)],
                dim = -1
            ) # (K, 3)
        v_ending_temp *= thickness # (K, 3)
        v_ending_temp = v_ending_temp[None].repeat(N, 1, 1) # (N, K, 3)

        # 2. Rotate the template plane to the direction of the line.
        rot_axis = torch.cross(vec_y, dir) # (N, 3)
        rot_axis[neg_mask] *= -1
        rot_mat = transforms.axis_angle_to_matrix(rot_axis) # (N, 3, 3)
        v_ending_temp = v_ending_temp @ rot_mat.transpose(-1, -2)
        v_ending_temp = v_ending_temp.to(device)

        # 3. Move the template plane to the start and end points and get the cylinder vertices.
        v_cylinder_start = v_ending_temp + start_points[:, None] # (N, K, 3)
        v_cylinder_end = v_ending_temp + end_points[:, None] # (N, K, 3)
        #    Swap the start and end points for the negative direction to adjust the normal direction.
        v_cylinder_start[neg_mask], v_cylinder_end[neg_mask] = v_cylinder_end[neg_mask], v_cylinder_start[neg_mask]
        v_cylinder = torch.cat([v_cylinder_start, v_cylinder_end], dim=1) # (N, 2*K, 3)

        # ===== Calculate the face index. =====
        idx = torch.arange(0, 2*K, device=device).to(device) # (2*K,)
        idx_s, idx_e = idx[:K], idx[K:]
        f_cylinder = torch.cat([
            # Two ending surface.
            torch.stack([idx_s[0].repeat(K-2), idx_s[1:-1], idx_s[2:]], dim=-1),
            torch.stack([idx_e[0].repeat(K-2), idx_e[2:], idx_e[1:-1]], dim=-1),
            # The side surface.
            torch.stack([idx_e[:-1], idx_s[1:], idx_s[:-1]], dim=-1),
            torch.stack([idx_e[:-1], idx_e[1:], idx_s[1:]], dim=-1),
        ], dim=0) # (4*K-4, 3)
        f_cylinder = f_cylinder[None].repeat(N, 1, 1) # (N, 4*K-4, 3)

        # ===== Calculate the face index. =====
        if colors is not None:
            c_cylinder = colors / 255.0 # (N, 3)
            c_cylinder = c_cylinder[:, None].repeat(1, 2*K, 1) # (N, 2*K, 3)
        else:
            c_cylinder = None

        N, V = v_cylinder.shape[:2]
        v_cylinder = v_cylinder.reshape(-1, 3) # (N*(2*K), 3)


        # ===== Manually match the points index before flatten. =====
        f_cylinder = f_cylinder + torch.arange(0, N, device=device).unsqueeze(1).unsqueeze(1) * V
        f_cylinder = f_cylinder.reshape(-1, 3) # (N*(4*K-4), 3)
        if c_cylinder is not None:
            c_cylinder = c_cylinder.reshape(-1, 3) # (N*(2*K), 3)

        self.add_mesh(
            vertices      = v_cylinder,
            vertex_colors = c_cylinder,
            faces         = f_cylinder,
            name          = name,
        )