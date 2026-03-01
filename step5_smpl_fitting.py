"""
Step 5: SMPL Model Fitting
============================
Reads: step-4-out/ (joints_3d_stereo) + step-3-out/ (smpl_indices)
Writes: step-5-out/ (fitted_joints, fitted_verts, faces, betas, pose, trans)
"""
import os
import sys
import argparse
import pickle
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_logger import StepLogger
from constants import SMPL_JOINT_NAMES, SMPL_MODEL_PATH


# ============================================================
# SMPLModel — fully contained in this step
# ============================================================
class SMPLModel:
    """Minimal SMPL model loader and forward pass using PyTorch."""

    def __init__(self, model_path, device='cuda'):
        self.device = device
        print(f"Loading SMPL model from {model_path}...")

        # Stub to replace chumpy objects during unpickling
        class _ChumpyStub:
            def __init__(self, *args, **kwargs):
                pass
            def __setstate__(self, state):
                if isinstance(state, dict) and 'x' in state:
                    self._data = np.array(state['x'])
                elif isinstance(state, np.ndarray):
                    self._data = state
                else:
                    self._data = None
            def __array__(self):
                if hasattr(self, '_data') and self._data is not None:
                    return self._data
                return np.array([])

        import types
        fake_ch = types.ModuleType('chumpy')
        fake_ch.Ch = _ChumpyStub
        fake_ch.array = _ChumpyStub
        fake_ch_linalg = types.ModuleType('chumpy.linalg')
        fake_ch_utils = types.ModuleType('chumpy.utils')
        sys.modules['chumpy'] = fake_ch
        sys.modules['chumpy.ch'] = fake_ch
        sys.modules['chumpy.linalg'] = fake_ch_linalg
        sys.modules['chumpy.utils'] = fake_ch_utils
        for attr in ['Ch', 'array', 'MatVecMult', 'Select', 'Rodrigues',
                     'dot', 'multiply', 'subtract', 'add']:
            setattr(fake_ch, attr, _ChumpyStub)
            setattr(fake_ch_linalg, attr, _ChumpyStub)

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f, encoding='latin1')

        # Convert chumpy stubs to numpy
        for k, v in model_data.items():
            if isinstance(v, _ChumpyStub):
                model_data[k] = np.array(v)
            elif hasattr(v, 'r'):
                model_data[k] = np.array(v.r)

        # Extract model components
        self.v_template = torch.tensor(
            np.array(model_data['v_template']), dtype=torch.float32, device=device
        )
        self.shapedirs = torch.tensor(
            np.array(model_data['shapedirs'])[:, :, :10], dtype=torch.float32, device=device
        )
        self.posedirs = torch.tensor(
            np.array(model_data['posedirs']), dtype=torch.float32, device=device
        )
        self.J_regressor = torch.tensor(
            np.array(model_data['J_regressor'].todense()), dtype=torch.float32, device=device
        )
        self.weights = torch.tensor(
            np.array(model_data['weights']), dtype=torch.float32, device=device
        )

        self.kintree_table = np.array(model_data['kintree_table']).astype(np.int64)
        self.faces = np.array(model_data['f']).astype(np.int32)

        self.n_joints = 24
        self.n_verts = 6890
        self.n_betas = 10

        self.parent = {
            i: int(self.kintree_table[0, i])
            for i in range(1, self.kintree_table.shape[1])
        }

        print(f"  Vertices: {self.n_verts}, Joints: {self.n_joints}, Faces: {len(self.faces)}")
        print("  SMPL model loaded successfully.")

    def rodrigues(self, r):
        """Convert axis-angle to rotation matrix (batch)."""
        theta = torch.norm(r, dim=1, keepdim=True).unsqueeze(-1)
        r_hat = r / (torch.norm(r, dim=1, keepdim=True) + 1e-8)

        cos = torch.cos(theta)
        sin = torch.sin(theta)

        K = torch.zeros(r.shape[0], 3, 3, device=r.device, dtype=r.dtype)
        K[:, 0, 1] = -r_hat[:, 2]
        K[:, 0, 2] = r_hat[:, 1]
        K[:, 1, 0] = r_hat[:, 2]
        K[:, 1, 2] = -r_hat[:, 0]
        K[:, 2, 0] = -r_hat[:, 1]
        K[:, 2, 1] = r_hat[:, 0]

        I = torch.eye(3, device=r.device, dtype=r.dtype).unsqueeze(0)
        R = I + sin * K + (1 - cos) * torch.bmm(K, K)
        return R

    def forward(self, betas=None, pose=None, trans=None):
        """Run SMPL forward pass. Returns: vertices (6890, 3), joints (24, 3)"""
        if betas is None:
            betas = torch.zeros(self.n_betas, device=self.device)
        if pose is None:
            pose = torch.zeros(self.n_joints * 3, device=self.device)
        if trans is None:
            trans = torch.zeros(3, device=self.device)

        v_shaped = self.v_template + torch.einsum('ijk,k->ij', self.shapedirs, betas)
        J = torch.matmul(self.J_regressor, v_shaped)

        pose_params = pose.reshape(-1, 3)
        rot_mats = self.rodrigues(pose_params)

        ident = torch.eye(3, device=self.device).unsqueeze(0)
        pose_feature = (rot_mats[1:] - ident).reshape(-1)
        v_posed = v_shaped + torch.einsum('ijk,k->ij', self.posedirs, pose_feature)

        G = [self._make_transform(rot_mats[0], J[0])]
        for i in range(1, self.n_joints):
            parent_idx = self.parent[i]
            local_t = self._make_transform(rot_mats[i], J[i] - J[parent_idx])
            G.append(torch.matmul(G[parent_idx], local_t))
        G = torch.stack(G)

        posed_joints = G[:, :3, 3].clone()

        J_homo = torch.cat([J, torch.zeros(self.n_joints, 1, device=self.device)], dim=1)
        G_offset = G.clone()
        for i in range(self.n_joints):
            rest_in_world = torch.matmul(G[i], J_homo[i])
            G_offset[i, :3, 3] = G[i, :3, 3] - rest_in_world[:3]

        G_flat = G_offset.reshape(self.n_joints, 16)
        T = torch.matmul(self.weights, G_flat).reshape(self.n_verts, 4, 4)

        v_homo = torch.cat([v_posed, torch.ones(self.n_verts, 1, device=self.device)], dim=1)
        v_final = torch.einsum('nij,nj->ni', T, v_homo)[:, :3]

        v_final = v_final + trans
        posed_joints = posed_joints + trans

        return v_final, posed_joints

    def _make_transform(self, R, t):
        T = torch.zeros(4, 4, device=self.device)
        T[:3, :3] = R
        T[:3, 3] = t
        T[3, 3] = 1.0
        return T

    def fit_to_joints(self, target_joints_3d, joint_indices, n_iters=500, lr=0.01):
        """Fit SMPL parameters to target 3D joints."""
        target = torch.tensor(target_joints_3d, dtype=torch.float32, device=self.device)

        betas = torch.zeros(self.n_betas, device=self.device, requires_grad=True)
        pose = torch.zeros(self.n_joints * 3, device=self.device, requires_grad=True)
        trans = torch.tensor(
            target_joints_3d.mean(axis=0), dtype=torch.float32, device=self.device,
            requires_grad=True
        )

        optimizer = torch.optim.Adam([
            {'params': [trans], 'lr': lr * 10},
            {'params': [betas], 'lr': lr},
            {'params': [pose], 'lr': lr * 0.5},
        ])

        print(f"\n  Fitting SMPL to {len(joint_indices)} joints ({n_iters} iters)...")

        best_loss = float('inf')
        best_params = None

        for i in range(n_iters):
            optimizer.zero_grad()

            verts, joints = self.forward(betas, pose, trans)

            pred_joints = joints[joint_indices]
            loss_joints = torch.mean((pred_joints - target) ** 2)

            loss_betas = 0.0001 * torch.sum(betas ** 2)
            loss_pose = 0.0001 * torch.sum(pose ** 2)

            loss = loss_joints + loss_betas + loss_pose

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = (
                    betas.detach().clone(),
                    pose.detach().clone(),
                    trans.detach().clone()
                )

            if (i + 1) % 100 == 0:
                joint_err = torch.sqrt(loss_joints).item()
                print(f"    Iter {i+1}/{n_iters}: loss={loss.item():.6f}, joint_RMSE={joint_err*1000:.1f}mm")

        betas_f, pose_f, trans_f = best_params
        with torch.no_grad():
            verts_final, joints_final = self.forward(betas_f, pose_f, trans_f)

        print(f"  Fitting complete. Final loss: {best_loss:.6f}")

        return (
            betas_f.cpu().numpy(),
            pose_f.cpu().numpy(),
            trans_f.cpu().numpy(),
            verts_final.cpu().numpy(),
            joints_final.cpu().numpy()
        )


def run(logger, smpl_path=SMPL_MODEL_PATH, known_height=None,
        joints_3d_stereo=None, smpl_indices=None):
    """Fit SMPL model to triangulated 3D joints."""
    print("\n[5/7] Fitting SMPL model")

    # Load from previous steps if not provided
    if joints_3d_stereo is None:
        step4 = logger.load_step_output(4)
        joints_3d_stereo = np.array(step4["joints_3d_stereo"])

    if smpl_indices is None:
        step3 = logger.load_step_output(3)
        smpl_indices = step3["smpl_indices"]

    smpl = SMPLModel(smpl_path, device='cpu')

    # Convert stereo coords to SMPL space (mm -> m, flip Y)
    target_joints = joints_3d_stereo.copy().astype(np.float64)
    target_joints /= 1000.0  # mm -> m
    target_joints[:, 1] *= -1  # camera Y-down -> SMPL Y-up
    print(f"  Converted mm→m and flipped Y (camera Y-down → SMPL Y-up)")

    # Fit
    betas, pose, trans, fitted_verts, fitted_joints = smpl.fit_to_joints(
        target_joints, smpl_indices, n_iters=1000, lr=0.01
    )

    # Convert back to mm
    fitted_verts_mm = (fitted_verts * 1000.0).astype(np.float64)
    fitted_verts_mm[:, 1] *= -1
    fitted_joints_mm = (fitted_joints * 1000.0).astype(np.float64)
    fitted_joints_mm[:, 1] *= -1

    print(f"\n  SMPL shape parameters (betas): {betas[:5]}...")
    print(f"  SMPL fitted {len(fitted_joints_mm)} joints, {len(fitted_verts_mm)} vertices")

    logger.log_step(5, "in", {
        "smpl_path": smpl_path,
        "num_target_joints": len(joints_3d_stereo),
    })
    logger.log_step(5, "out", {
        "fitted_joints": fitted_joints_mm,
        "fitted_verts": fitted_verts_mm,
    })

    # Save additional arrays for later steps
    out_dir = os.path.join(logger.run_dir, "step-5-out")
    np.save(os.path.join(out_dir, "fitted_verts.npy"), fitted_verts_mm)
    np.save(os.path.join(out_dir, "betas.npy"), betas)
    np.save(os.path.join(out_dir, "pose.npy"), pose)
    np.save(os.path.join(out_dir, "trans.npy"), trans)
    np.save(os.path.join(out_dir, "faces.npy"), smpl.faces)

    # Apply known height scaling if provided
    if known_height is not None:
        from constants import HEIGHT_PATH
        total = 0.0
        for i in range(len(HEIGHT_PATH) - 1):
            i1 = SMPL_JOINT_NAMES.index(HEIGHT_PATH[i])
            i2 = SMPL_JOINT_NAMES.index(HEIGHT_PATH[i + 1])
            total += float(np.linalg.norm(fitted_joints_mm[i1] - fitted_joints_mm[i2]))
        raw_height_cm = total / 10.0
        scale = known_height / raw_height_cm
        fitted_joints_mm *= scale
        fitted_verts_mm *= scale
        print(f"  Height scaling: {raw_height_cm:.1f}cm → {known_height:.1f}cm (scale={scale:.4f})")

    return fitted_joints_mm, fitted_verts_mm, smpl_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: SMPL model fitting")
    parser.add_argument("--run-dir", required=True, help="Run directory from step 4")
    parser.add_argument("--smpl", default=SMPL_MODEL_PATH, help="SMPL model path")
    parser.add_argument("--height", type=float, default=None, help="Known height in cm")
    args = parser.parse_args()

    logger = StepLogger.from_run_dir(args.run_dir)
    run(logger, smpl_path=args.smpl, known_height=args.height)
    print(f"  ✓ Step 5 complete. Run dir: {logger.run_dir}")
