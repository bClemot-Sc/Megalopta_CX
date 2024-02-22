def CIU_activation(heading_direction):
    relative_heading = (-heading_direction) % 360
    heading_list = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    closest_heading = min(heading_list, key=lambda x: abs(x - relative_heading))
    heading_id = heading_list.index(closest_heading % 360) + 1
    return str(heading_id)

print(CIU_activation(135))