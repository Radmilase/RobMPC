"""
Parses the MuJoCo XML file to extract maze wall geometries.
"""
import xml.etree.ElementTree as ET
import numpy as np
from maze_config import MAZE_XML_PATH

class Wall:
    """Represents a rectangular wall in 2D space."""
    def __init__(self, center_x, center_y, size_x, size_y, name=None):
        """
        Args:
            center_x (float): X-coordinate of the wall's center.
            center_y (float): Y-coordinate of the wall's center.
            size_x (float): Half-size of the wall along its local X-axis (width/2).
            size_y (float): Half-size of the wall along its local Y-axis (depth/2).
                           Note: MuJoCo box geoms size are half-lengths.
            name (str, optional): Name of the geom.
        """
        self.center = np.array([center_x, center_y])
        # Store half-sizes directly as they are given in MuJoCo XML for 'box' type
        self.half_size_x = size_x 
        self.half_size_y = size_y
        self.name = name

        # Calculate corner points for easier collision checking or visualization
        # Assumes walls are axis-aligned in their local frame, which is true for non-rotated top-level geoms
        self.min_x = center_x - size_x
        self.max_x = center_x + size_x
        self.min_y = center_y - size_y
        self.max_y = center_y + size_y

    def __repr__(self):
        return f"Wall(name='{self.name}', center=({self.center[0]:.2f}, {self.center[1]:.2f}), half_sizes=({self.half_size_x:.2f}, {self.half_size_y:.2f}))"

    def get_patch_params(self):
        """Returns parameters океfor a Matplotlib Rectangle patch."""
        return (self.min_x, self.min_y), 2 * self.half_size_x, 2 * self.half_size_y

class MazeParser:
    """Parses a MuJoCo XML file to extract maze wall information."""
    def __init__(self, xml_path=MAZE_XML_PATH):
        self.xml_path = xml_path
        self.walls = []
        self.world_min_bounds = np.array([np.inf, np.inf])
        self.world_max_bounds = np.array([-np.inf, -np.inf])
        self._parse_xml()

    def _parse_xml(self):
        """Protected method to parse the XML and populate walls."""
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
        except FileNotFoundError:
            print(f"Error: XML file not found at {self.xml_path}")
            return
        except ET.ParseError as e:
            print(f"Error parsing XML file {self.xml_path}: {e}")
            return

        # Look for geoms within worldbody or nested bodies
        # For scene_2.xml, walls are directly under a body in worldbody
        for body_element in root.findall('.//worldbody/body'): # Finds all <body> elements under <worldbody>
            # Get body position offset, default to (0,0,0) if not specified
            body_pos_str = body_element.get('pos', '0 0 0')
            body_pos_parts = list(map(float, body_pos_str.split()))
            body_offset = np.array([body_pos_parts[0], body_pos_parts[1]]) # Only X, Y

            for geom_element in body_element.findall('geom'):
                if geom_element.get('type') == 'box':
                    name = geom_element.get('name', 'unnamed_wall')
                    pos_str = geom_element.get('pos', '0 0 0')
                    size_str = geom_element.get('size', '0 0 0') # half-lengths
                    # rgba_str = geom_element.get('rgba') # For potential filtering if needed

                    try:
                        pos_parts = list(map(float, pos_str.split()))
                        size_parts = list(map(float, size_str.split()))
                    except ValueError as e:
                        print(f"Warning: Could not parse pos/size for geom '{name}': {e}. Skipping.")
                        continue

                    # We are interested in 2D plane (X, Y)
                    # Geom position is relative to its parent body
                    center_x = body_offset[0] + pos_parts[0]
                    center_y = body_offset[1] + pos_parts[1]
                    # For a box, size is (half_width, half_depth, half_height)
                    # We assume walls are primarily defined in XY plane, so size_x, size_y are relevant
                    half_size_x = size_parts[0]
                    half_size_y = size_parts[1]
                    # We ignore Z-dimension for 2D path planning

                    wall = Wall(center_x, center_y, half_size_x, half_size_y, name)
                    self.walls.append(wall)
                    
                    # Update world bounds
                    self.world_min_bounds[0] = min(self.world_min_bounds[0], wall.min_x)
                    self.world_min_bounds[1] = min(self.world_min_bounds[1], wall.min_y)
                    self.world_max_bounds[0] = max(self.world_max_bounds[0], wall.max_x)
                    self.world_max_bounds[1] = max(self.world_max_bounds[1], wall.max_y)

        print(f"Parsed {len(self.walls)} walls from {self.xml_path}.")
        if not self.walls:
            print("Warning: No walls were parsed. Check XML structure and geom types.")
        else:
            print(f"World bounds (min_x, min_y): ({self.world_min_bounds[0]:.2f}, {self.world_min_bounds[1]:.2f})")
            print(f"World bounds (max_x, max_y): ({self.world_max_bounds[0]:.2f}, {self.world_max_bounds[1]:.2f})")

    def get_walls(self):
        """Returns the list of parsed Wall objects."""
        return self.walls

    def get_world_bounds(self):
        """Returns the min and max XY coordinates of the parsed world."""
        return self.world_min_bounds, self.world_max_bounds

if __name__ == '__main__':
    # Example usage:
    parser = MazeParser()
    walls = parser.get_walls()
    for wall in walls:
        print(wall)
    min_bounds, max_bounds = parser.get_world_bounds()
    # print(f"Min bounds: {min_bounds}, Max bounds: {max_bounds}") 