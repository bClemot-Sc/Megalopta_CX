def compare_headings(previous_heading, new_heading):
    # Normalize headings to be between 0 and 360 degrees
    previous_heading %= 360
    new_heading %= 360

    # Calculate the difference between the new and previous headings
    heading_difference = (new_heading - previous_heading) % 360

    # Determine the direction of the turn (left, right, or not turning)
    if heading_difference == 0:
        turn_direction = "not turning"
    elif heading_difference <= 180:
        turn_direction = "left"
    else:
        turn_direction = "right"

    return turn_direction

# Example usage:
previous_heading = 350  # Replace with your actual previous heading
new_heading = 340      # Replace with your actual new heading

turn_direction = compare_headings(previous_heading, new_heading)
print(f"The object is {turn_direction}.")
