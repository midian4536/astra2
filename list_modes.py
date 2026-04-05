from pyorbbecsdk import *

pipeline = Pipeline()

depth_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
print("depth profile count:", len(depth_profiles))

for i in range(len(depth_profiles)):
    p = depth_profiles[i]
    try:
        print(
            i,
            "w=", p.get_width(),
            "h=", p.get_height(),
            "fmt=", p.get_format(),
            "fps=", p.get_fps()
        )
    except Exception as e:
        print(i, p, e)