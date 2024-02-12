def adjust_orientation(angle):
    return angle % 360

def CIU_activation(heading_direction):
    heading_list = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    closest_heading = min(heading_list, key=lambda x: abs(x - heading_direction))
    heading_id = heading_list.index(adjust_orientation(closest_heading)) + 1
    return str(heading_id)


print("CIU" + CIU_activation(329))