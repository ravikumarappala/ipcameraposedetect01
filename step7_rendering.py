"""
Step 7: Mesh Rendering & Summary
==================================
Reads: step-5-out/ (fitted_joints, fitted_verts, faces.npy)
     + step-6-out/ (measurements)
     + step-2-out/ (stereo_info)
     + step-3-out/ (smpl_indices)
     + step-4-out/ (joints_3d_stereo)
Writes: step-7-out/ (annotated_mesh, mesh_image) + summary.csv, summary.json
"""
import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_logger import StepLogger
from constants import SMPL_JOINT_NAMES, SKELETON_CONNECTIONS, SEGMENT_MEASUREMENTS


# ============================================================
# Rendering functions — fully contained in this step
# ============================================================

def rotate_for_upright_view(verts_mm, center=None):
    """Rotate vertices so the body stands upright for rendering."""
    v = verts_mm.copy().astype(np.float32)
    if center is None:
        center = v.mean(axis=0)
    v -= center
    x, y, z = v[:, 0].copy(), v[:, 1].copy(), v[:, 2].copy()
    # Camera coords: Y-down, Z=depth (varies little across body)
    # We want: X = left-right, Y = depth (≈0 after centering), Z = UP (body height)
    # Camera Y-down → negate so head goes to positive Z (up in matplotlib 3D)
    v[:, 0] = x   # keep left-right
    v[:, 1] = z   # new Y = depth (near-zero after centering)
    v[:, 2] = -y  # new Z = -camera_Y_down → positive = UP (head at top)
    return v


def render_smpl_mesh(verts_mm, faces, img_size=(800, 800)):
    """Render SMPL mesh with pyrender (offscreen)."""
    import trimesh
    import pyrender

    verts = verts_mm.astype(np.float32) / 1000.0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3, 1.0])

    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(mesh_node)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0.0, -2.0, 1.5]
    scene.add(camera, pose=cam_pose)

    light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light_pose1 = np.eye(4)
    light_pose1[:3, 3] = [0.0, -1.0, 2.0]
    scene.add(light1, pose=light_pose1)

    light2 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    light_pose2 = np.eye(4)
    light_pose2[:3, 3] = [1.0, -1.0, 1.0]
    scene.add(light2, pose=light_pose2)

    r = pyrender.OffscreenRenderer(img_size[0], img_size[1])
    color, _ = r.render(scene)
    r.delete()

    # Check for blank output (pyrender offscreen fails on some systems)
    if np.mean(color < 250) < 0.05:
        raise RuntimeError("pyrender produced blank image (offscreen rendering not supported)")

    return color


def render_smpl_mesh_matplotlib(verts_mm, faces, img_size=(800, 800)):
    """Fallback renderer using matplotlib (no OpenGL)."""
    v = verts_mm / 1000.0

    fig = plt.figure(figsize=(img_size[0] / 100.0, img_size[1] / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    tris = v[faces]
    mesh = Poly3DCollection(tris, alpha=0.8)
    mesh.set_facecolor((0.7, 0.7, 0.9))
    mesh.set_edgecolor((0.1, 0.1, 0.1))
    ax.add_collection3d(mesh)

    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    # Per-axis limits (Y/depth is much thinner than X/Z after upright rotation)
    pad = 0.15
    x_half = (x.max() - x.min()) / 2.0 * (1 + pad)
    y_half = max((y.max() - y.min()) / 2.0, x_half * 0.3) * (1 + pad)
    z_half = (z.max() - z.min()) / 2.0 * (1 + pad)
    mid_x = (x.max() + x.min()) / 2.0
    mid_y = (y.max() + y.min()) / 2.0
    mid_z = (z.max() + z.min()) / 2.0

    ax.set_xlim(mid_x - x_half, mid_x + x_half)
    ax.set_ylim(mid_y - y_half, mid_y + y_half)
    ax.set_zlim(mid_z - z_half, mid_z + z_half)

    ax.view_init(elev=5, azim=90)  # front view: looking along depth (Y) axis, Z=up
    ax.axis("off")

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf).reshape(h, w, 4)[:, :, :3].copy()
    plt.close(fig)
    return img


def render_smpl_mesh_annotated(verts_mm, faces, joints_mm, measurements, img_size=(1200, 1000)):
    """
    Render 3D SMPL mesh with joints, skeleton, lengths, and angles.
    Tries pyrender first; falls back to matplotlib if blank output detected.
    Annotations (joint names, lengths, angles) overlaid via cv2.
    """
    verts_m = verts_mm.astype(np.float64) / 1000.0
    joints_m = joints_mm.astype(np.float64) / 1000.0

    img = None
    joints_2d = None
    renderer_used = None

    # ============ Try pyrender ============
    try:
        import trimesh
        import pyrender

        scene = pyrender.Scene(bg_color=[255, 255, 255, 255],
                               ambient_light=[0.4, 0.4, 0.4, 1.0])

        # Body mesh (semi-transparent)
        body_mesh = trimesh.Trimesh(vertices=verts_m, faces=faces, process=False)
        body_mesh.fix_normals()
        body_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.65, 0.72, 0.88, 0.55],
            metallicFactor=0.1, roughnessFactor=0.6,
            alphaMode='BLEND'
        )
        scene.add(pyrender.Mesh.from_trimesh(body_mesh, material=body_material, smooth=True))

        # Joint spheres
        joint_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.9, 0.15, 0.15, 1.0],
            metallicFactor=0.3, roughnessFactor=0.4
        )
        for idx in range(len(joints_m)):
            sphere = trimesh.creation.uv_sphere(radius=0.015, count=[8, 8])
            sphere.apply_translation(joints_m[idx])
            scene.add(pyrender.Mesh.from_trimesh(sphere, material=joint_material, smooth=True))

        # Skeleton bones
        bone_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.95, 0.3, 0.3, 1.0],
            metallicFactor=0.2, roughnessFactor=0.5
        )
        for i_conn, j_conn in SKELETON_CONNECTIONS:
            if i_conn < len(joints_m) and j_conn < len(joints_m):
                p1, p2 = joints_m[i_conn], joints_m[j_conn]
                seg = p2 - p1
                length = np.linalg.norm(seg)
                if length < 1e-6:
                    continue
                cyl = trimesh.creation.cylinder(radius=0.006, height=length, sections=8)
                seg_dir = seg / length
                z_axis = np.array([0.0, 0.0, 1.0])
                cr = np.cross(z_axis, seg_dir)
                dt = np.dot(z_axis, seg_dir)
                if np.linalg.norm(cr) < 1e-6:
                    R = np.eye(3) if dt > 0 else np.diag([1.0, -1.0, -1.0])
                else:
                    cr_n = cr / np.linalg.norm(cr)
                    ang = np.arccos(np.clip(dt, -1, 1))
                    K = np.array([[0, -cr_n[2], cr_n[1]], [cr_n[2], 0, -cr_n[0]], [-cr_n[1], cr_n[0], 0]])
                    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
                T_mat = np.eye(4)
                T_mat[:3, :3] = R
                T_mat[:3, 3] = (p1 + p2) / 2.0
                cyl.apply_transform(T_mat)
                scene.add(pyrender.Mesh.from_trimesh(cyl, material=bone_material, smooth=True))

        # Camera
        v_center = verts_m.mean(axis=0)
        v_extent = (verts_m.max(axis=0) - verts_m.min(axis=0)).max()
        yfov = np.pi / 6.0
        cam_dist = v_extent / (2.0 * np.tan(yfov / 2.0)) * 1.4
        camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=img_size[0] / img_size[1])
        cam_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        cam_t = v_center + np.array([0.0, -cam_dist, 0.0])
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = cam_R
        cam_pose[:3, 3] = cam_t
        scene.add(camera, pose=cam_pose)

        # Lights
        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(dl, pose=cam_pose)

        r = pyrender.OffscreenRenderer(img_size[0], img_size[1])
        color_img, _ = r.render(scene)
        r.delete()

        # Check if the render is blank (all white = failed)
        non_white = np.mean(color_img < 250)
        if non_white > 0.05:
            img = color_img[:, :, :3].copy() if color_img.shape[2] == 4 else color_img.copy()
            fy = img_size[1] / (2.0 * np.tan(yfov / 2.0))
            fx = fy
            cx, cy_c = img_size[0] / 2.0, img_size[1] / 2.0
            cam_inv = np.linalg.inv(cam_pose)
            j_cam = (cam_inv[:3, :3] @ joints_m.T + cam_inv[:3, 3:4]).T
            joints_2d = []
            for jc in j_cam:
                px = int(fx * jc[0] / (-jc[2]) + cx) if abs(jc[2]) > 1e-6 else int(cx)
                py = int(fy * (-jc[1]) / (-jc[2]) + cy_c) if abs(jc[2]) > 1e-6 else int(cy_c)
                joints_2d.append((px, py))
            renderer_used = 'pyrender'
            print("    Using pyrender renderer")
        else:
            print("    pyrender produced blank image, falling back to matplotlib")

    except Exception as e:
        print(f"    pyrender error: {e}, falling back to matplotlib")

    # ============ Matplotlib fallback ============
    if img is None:
        from mpl_toolkits.mplot3d import proj3d

        v = verts_m
        j = joints_m

        fig = plt.figure(figsize=(img_size[0] / 100.0, img_size[1] / 100.0), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Lambert-shaded mesh
        tris = v[faces]
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        fn = np.cross(v1 - v0, v2 - v0)
        fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
        light = np.array([0.2, -0.8, 0.5])
        light = light / np.linalg.norm(light)
        diff = np.abs(np.dot(fn, light))
        intensity = np.clip(0.35 + 0.65 * diff, 0, 1)
        base_c = np.array([0.65, 0.72, 0.88])
        fc = np.outer(intensity, base_c)
        fc = np.clip(fc, 0, 1)
        fc_rgba = np.column_stack([fc, np.ones(len(fc)) * 0.6])

        sort_idx = np.argsort(tris[:, :, 1].mean(axis=1))
        mesh_coll = Poly3DCollection(tris[sort_idx])
        mesh_coll.set_facecolor(fc_rgba[sort_idx])
        mesh_coll.set_edgecolor('none')
        ax.add_collection3d(mesh_coll)

        # Skeleton in 3D
        for i_c, j_c in SKELETON_CONNECTIONS:
            if i_c < len(j) and j_c < len(j):
                ax.plot([j[i_c,0], j[j_c,0]], [j[i_c,1], j[j_c,1]], [j[i_c,2], j[j_c,2]],
                        '-', color='#FF3333', linewidth=2.5, zorder=10)
        for idx in range(len(j)):
            ax.scatter(j[idx,0], j[idx,1], j[idx,2], c='#CC0000', s=50,
                       zorder=15, depthshade=False, edgecolors='white', linewidths=0.8)

        xv, yv, zv = v[:,0], v[:,1], v[:,2]
        pad = 0.15
        x_half = (xv.max() - xv.min()) / 2.0 * (1 + pad)
        y_half = max((yv.max() - yv.min()) / 2.0, x_half * 0.3) * (1 + pad)
        z_half = (zv.max() - zv.min()) / 2.0 * (1 + pad)
        mid = np.array([(xv.max()+xv.min())/2, (yv.max()+yv.min())/2, (zv.max()+zv.min())/2])
        ax.set_xlim(mid[0]-x_half, mid[0]+x_half)
        ax.set_ylim(mid[1]-y_half, mid[1]+y_half)
        ax.set_zlim(mid[2]-z_half, mid[2]+z_half)
        ax.view_init(elev=5, azim=90)  # front view: looking along depth (Y) axis, Z=up
        ax.axis('off')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        fig.canvas.draw()

        # Project joints to 2D
        joints_2d = []
        for idx in range(len(j)):
            x2, y2, _ = proj3d.proj_transform(j[idx,0], j[idx,1], j[idx,2], ax.get_proj())
            xp, yp = ax.transData.transform((x2, y2))
            joints_2d.append((int(round(xp)), int(round(img_size[1] - yp))))

        w_f, h_f = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf).reshape(h_f, w_f, 4)[:, :, :3].copy()
        plt.close(fig)
        renderer_used = 'matplotlib'
        print("    Using matplotlib renderer")

    # ============ Overlay 2D annotations with cv2 ============
    key_joints = ['head', 'neck', 'l_shoulder', 'r_shoulder',
                  'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
                  'pelvis', 'l_hip', 'r_hip',
                  'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    for idx in range(min(len(joints_2d), 24)):
        if SMPL_JOINT_NAMES[idx] in key_joints:
            pt = joints_2d[idx]
            cv2.putText(img, SMPL_JOINT_NAMES[idx],
                        (pt[0] + 12, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)

    seg_labels = [
        ("Shoulder Width", 16, 17), ("Left Upper Arm", 16, 18),
        ("Left Forearm", 18, 20), ("Right Upper Arm", 17, 19),
        ("Right Forearm", 19, 21), ("Left Thigh", 1, 4),
        ("Left Shin", 4, 7), ("Right Thigh", 2, 5),
        ("Right Shin", 5, 8), ("Torso (Pelvis to Neck)", 0, 12),
        ("Hip Width", 1, 2),
    ]
    for seg_name, j1, j2 in seg_labels:
        if seg_name in measurements:
            mx = (joints_2d[j1][0] + joints_2d[j2][0]) // 2
            my = (joints_2d[j1][1] + joints_2d[j2][1]) // 2
            label = f"{measurements[seg_name]['cm']:.1f}cm"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (mx+5, my-th-4), (mx+tw+12, my+4), (255,255,255), -1)
            cv2.rectangle(img, (mx+5, my-th-4), (mx+tw+12, my+4), (180, 80, 0), 1)
            cv2.putText(img, label, (mx+8, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 80, 0), 1, cv2.LINE_AA)

    angle_defs = [
        ("L Elbow", 16, 18, 20), ("R Elbow", 17, 19, 21),
        ("L Knee", 1, 4, 7), ("R Knee", 2, 5, 8),
        ("L Shoulder", 12, 16, 18), ("R Shoulder", 12, 17, 19),
    ]
    for angle_name, ja, jb, jc in angle_defs:
        va_v = joints_m[ja] - joints_m[jb]
        vb_v = joints_m[jc] - joints_m[jb]
        cos_a = np.dot(va_v, vb_v) / (np.linalg.norm(va_v) * np.linalg.norm(vb_v) + 1e-8)
        angle_deg = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        pt = joints_2d[jb]
        label = f"{angle_name} {angle_deg:.0f} deg"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cv2.rectangle(img, (pt[0]-tw-14, pt[1]+6), (pt[0]-6, pt[1]+th+12), (230,240,255), -1)
        cv2.rectangle(img, (pt[0]-tw-14, pt[1]+6), (pt[0]-6, pt[1]+th+12), (0,100,200), 1)
        cv2.putText(img, label, (pt[0]-tw-10, pt[1]+th+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 100, 200), 1, cv2.LINE_AA)

    if 'Total Height (chain)' in measurements:
        h_cm = measurements['Total Height (chain)']['cm']
        top_y = min(p[1] for p in joints_2d)
        bot_y = max(p[1] for p in joints_2d)
        rx = max(p[0] for p in joints_2d) + 40
        rx = min(rx, img.shape[1] - 140)
        cv2.line(img, (rx, top_y), (rx, bot_y), (0, 153, 0), 2, cv2.LINE_AA)
        cv2.line(img, (rx-8, top_y), (rx+8, top_y), (0, 153, 0), 2)
        cv2.line(img, (rx-8, bot_y), (rx+8, bot_y), (0, 153, 0), 2)
        label = f"Height: {h_cm:.1f}cm"
        mid_ht = (top_y + bot_y) // 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (rx+6, mid_ht-th-4), (rx+tw+14, mid_ht+6), (232,245,232), -1)
        cv2.rectangle(img, (rx+6, mid_ht-th-4), (rx+tw+14, mid_ht+6), (0,153,0), 1)
        cv2.putText(img, label, (rx+10, mid_ht+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1, cv2.LINE_AA)

    cv2.putText(img, f"SMPL 3D Mesh with Joints, Lengths & Angles ({renderer_used})",
                (img.shape[1]//2 - 280, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

    return img


# ============================================================
# Step 7 run function
# ============================================================
def run(logger, known_height=None,
        fitted_joints=None, fitted_verts=None, measurements=None,
        stereo_info=None, joints_3d_stereo=None, smpl_indices=None):
    """Render SMPL mesh with annotations and write summary."""
    print("\n[7/7] Rendering SMPL mesh with joints, lengths & angles")

    # Load from previous steps if not provided
    if fitted_joints is None:
        step5 = logger.load_step_output(5)
        fitted_joints = np.array(step5["fitted_joints"])

    if fitted_verts is None:
        verts_path = os.path.join(logger.run_dir, "step-5-out", "fitted_verts.npy")
        if os.path.exists(verts_path):
            fitted_verts = np.load(verts_path)
        else:
            raise FileNotFoundError("fitted_verts.npy not found in step-5-out/")

    # Load faces from step-5-out (saved by step5 so we don't need SMPL model here)
    faces_path = os.path.join(logger.run_dir, "step-5-out", "faces.npy")
    if os.path.exists(faces_path):
        faces = np.load(faces_path)
    else:
        raise FileNotFoundError("faces.npy not found in step-5-out/. Re-run step 5.")

    if measurements is None:
        step6 = logger.load_step_output(6)
        measurements = step6["measurements"]

    if stereo_info is None:
        step2 = logger.load_step_output(2)
        stereo_info = step2.get("stereo_info", {"baseline_mm": 0, "img_size": [0, 0]})

    if joints_3d_stereo is None:
        step4 = logger.load_step_output(4)
        joints_3d_stereo = np.array(step4["joints_3d_stereo"])

    if smpl_indices is None:
        step3 = logger.load_step_output(3)
        smpl_indices = step3["smpl_indices"]

    logger.log_step(7, "in", {"measurements": measurements, "num_vertices": len(fitted_verts)})

    # Rotate for upright view
    shared_center = fitted_verts.mean(axis=0)
    verts_upright = rotate_for_upright_view(fitted_verts, center=shared_center)
    joints_upright = rotate_for_upright_view(fitted_joints, center=shared_center)

    # Annotated mesh render
    annotated_img = render_smpl_mesh_annotated(
        verts_upright, faces, joints_upright, measurements, img_size=(1200, 1000)
    )
    annotated_bgr = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

    # Plain mesh render
    try:
        mesh_img = render_smpl_mesh(verts_upright, faces, img_size=(800, 800))
        if mesh_img.shape[2] == 4:
            mesh_img_bgr = cv2.cvtColor(mesh_img, cv2.COLOR_RGBA2BGR)
        else:
            mesh_img_bgr = cv2.cvtColor(mesh_img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"  pyrender failed ({e}), using matplotlib.")
        mesh_img = render_smpl_mesh_matplotlib(verts_upright, faces, img_size=(800, 800))
        mesh_img_bgr = cv2.cvtColor(mesh_img, cv2.COLOR_RGB2BGR)

    logger.log_step(7, "out", {
        "annotated_mesh": annotated_bgr,
        "mesh_image": mesh_img_bgr,
    })

    # Write summary
    logger.write_summary(
        measurements=measurements,
        stereo_info=stereo_info,
        fitted_joints=fitted_joints,
        joints_3d_stereo=joints_3d_stereo,
        smpl_indices=smpl_indices,
        smpl_joint_names=SMPL_JOINT_NAMES,
        known_height=known_height,
    )

    return annotated_bgr, mesh_img_bgr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: Mesh rendering & summary")
    parser.add_argument("--run-dir", required=True, help="Run directory from step 6")
    parser.add_argument("--height", type=float, default=None, help="Known height in cm")
    args = parser.parse_args()

    logger = StepLogger.from_run_dir(args.run_dir)
    run(logger, known_height=args.height)
    print(f"  ✓ Step 7 complete. Run dir: {logger.run_dir}")
