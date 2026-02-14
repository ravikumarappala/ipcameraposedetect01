import numpy as np

#STEREO_PARAMS_FILE = "stereo_params_robust.npz"
STEREO_PARAMS_FILE = "stereo_params_sidebyside.npz"


def print_camera_parameters():
    """Load and print camera calibration parameters from stereo_params.npz"""
    try:
        data = np.load(STEREO_PARAMS_FILE)
        
        print("=" * 60)
        print("STEREO CAMERA PARAMETERS")
        print("=" * 60)
        
        # Print all available keys and their contents
        for key in sorted(data.files):
            print(f"\n{key}:")
            print("-" * 60)
            print(data[key])
            
        print("\n" + "=" * 60)
        
    except FileNotFoundError:
        print(f"ERROR: {STEREO_PARAMS_FILE} not found in current directory")
        return
    except Exception as e:
        print(f"ERROR: {e}")
        return


if __name__ == "__main__":
    print_camera_parameters()
