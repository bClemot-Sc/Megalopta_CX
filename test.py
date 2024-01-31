def adjust_orientation(angle):
    return angle % 360

def test_adjust_orientation():
    angles_to_test = [-720, -450, -360, -180, 0, 90, 180, 270, 360, 450, 720]

    for angle in angles_to_test:
        adjusted_angle = adjust_orientation(angle)
        print(f"Original Angle: {angle}, Adjusted Angle: {adjusted_angle}")

if __name__ == "__main__":
    test_adjust_orientation()