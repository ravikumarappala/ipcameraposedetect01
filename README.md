# Dual Camera Placement - Quick Run

Follow these steps to collect calibration data, compute stereo parameters, and run real-time arm tracking:

1. First run `capture_calibration_images90.py` to capture and store the images
2. Then run `stereo_calibration90.py` to calibrate and store the parameters
run [text](print_camera_params.py)
3. Then run `realtime_arm_3d.py` to track in realtime

/home/raviappala/archiconda3/envs/frfd01/bin/python /media/raviappala/edgeextvol2/jetson-inference/3dpose/traingulation/dual_Camera_placement/capture_calibration_images.py

home/raviappala/archiconda3/envs/frfd01/bin/python /media/raviappala/edgeextvol2/jetson-inference/3dpose/traingulation/dual_Camera_placement/stereo_calibration_90deg.py

To calibrate for accurate Human Body (X, Y, Z) joints, you need to calibrate the specific "volume" or space where the person will be standing.

1. Left/Right Only (Intrinsics)
Where to hold: Everywhere, especially the corners. Why: Lens distortion is worst at the edges of the image. Even if the person stands in the center, if they reach their hand to the side, that hand will be in the "distorted zone". If you don't calibrate the corners well, the hand's X,Y,Z will be wrong.

2. Combined / Shared (Extrinsics)
Where to hold: You need to "paint" the invisible box where the human will stand. Capture ~20-30 pairs at these specific spots:

Head Height: Hold rigid at ~1.7m (5.5ft) high.
Torso: Hold at chest level.
Feet: Put the board on the floor. (This is critical to get the ground plane right).
Arms Reach: Hold it far left and far right, where a person's hands might reach.
Depth: Do one set close to the cameras (1m) and one set further back (3-4m).
Summary: Don't just hold it in the center! The math only knows about the space where it "saw" the checkerboard. If you never showed it the floor, it won't be accurate at the feet.

Critically: Remember to STOP moving and wait 2 seconds for every single shared image.

0123:
we ran [capture_calibration_90deg_flip.py to capture iamges at 90degres - setup in hall with camras clamping on rods. We took 11 from the corner on each wall and also stood at 11ft in angle to cover both sides of the body and feet. took test image by running /media/raviappala/edgeextvol2/jetson-inference/3dpose/traingulation/dual_Camera_placement/capture_test_images.py
then we ran /media/raviappala/edgeextvol2/jetson-inference/3dpose/traingulation/dual_Camera_placement/stereo_calibration_90deg_modified.py to do calibration and output shown RMS - < 1 and used the same in hrnet to test and it gave a good results, except height of the body.>
after analysus suggesed to try keeping both cameras to the same side, so experimenting this setup

](capture_calibration_90deg_flip.py)# ipcameraposedetect01
