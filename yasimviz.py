#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import pyvista as pv
import os
import numpy as np
import argparse

POINT_LABEL_FONT_SIZE = 16       # Adjust point label font size to your preference
CAMERAMODE_FONT_SIZE = 10        # Adjust filename font size to your preference
THICKNESS_TO_CHORD_RATIO = 0.12  # Sets flight surface thickness to a typical 12%
WEIGHT_SPHERE_RADIUS = 0.20      # Radius of the weight item spheres

# Low eye strain and high visibility colour settings
COLORS = {
    'background': '#2e2e2e',  # Dark Gray
    'cameramode': '#8c9abf',  # Light Blue Gray
    'fuselage': '#8b8b8b',    # Medium Gray
    'wing': '#8c9abf',        # Light Blue Gray
    'hstab': '#9da3b3',       # Light Steel Blue
    'vstab': '#9e9e9e',       # Light Gray
    'mstab': '#a0a0a0',       # Gray
    'tank': '#6e8b8e',        # Slate Blue
    'ballast': '#ffb472',     # Peach
    'weight': '#ff7276'       # Light Coral
}

DEFAULTS = {
    "fuselages": {
        'ax': 0,
        'ay': 0,
        'az': 0,
        'bx': 0,
        'by': 0,
        'bz': 0,
        'width': 0,
        'taper': 1.0,
        'midpoint': 0.5},
    "wings": {
        'x': 0,
        'y': 0,
        'z': 0,
        'length': 0,
        'chord': 0,
        'taper': 1.0,
        'sweep': 0.0,
        'dihedral': 0.0,
        'incidence': 0.0,
        'twist': 0.0},
    "hstabs": {
        'x': 0,
        'y': 0,
        'z': 0,
        'length': 0,
        'chord': 0,
        'taper': 1.0,
        'sweep': 0.0,
        'dihedral': 0.0,
        'incidence': 0.0,
        'twist': 0.0},
    "vstabs": {
        'x': 0,
        'y': 0,
        'z': 0,
        'length': 0,
        'chord': 0,
        'taper': 1.0,
        'sweep': 0.0,
        'dihedral': 90.0,
        'incidence': 0.0,
        'twist': 0.0},
    "mstabs": {
        'x': 0,
        'y': 0,
        'z': 0,
        'length': 0,
        'chord': 0,
        'taper': 1.0,
        'sweep': 0.0,
        'dihedral': 0.0,
        'incidence': 0.0,
        'twist': 0.0},
    "tanks": {
        'x': 0,
        'y': 0,
        'z': 0,
        'capacity': 0},
    "ballasts": {
        'x': 0,
        'y': 0,
        'z': 0,
        'mass': 0},
    "weights": {
        'x': 0,
        'y': 0,
        'z': 0,
        'mass-prop': 'N/A'}}


def parse_yasim_file(file_path):
    def parse_component(element, attributes):
        parsed = {}
        for attr, default in attributes.items():
            value = element.get(attr, default)

            if attr == 'mass-prop':
                parsed[attr] = value if value is not None else default
            else:
                if value not in (None, 'N/A'):
                    try:
                        parsed[attr] = float(value)
                    except ValueError:
                        parsed[attr] = default
                else:
                    parsed[attr] = default

        return parsed

    tree = ET.parse(file_path)
    root = tree.getroot()

    components = {key: [] for key in DEFAULTS.keys()}

    for component_type, attributes in DEFAULTS.items():
        for element in root.findall(component_type[:-1]):
            component = parse_component(element, attributes)
            components[component_type].append(component)
            if component_type in ['wings', 'hstabs', 'mstabs']:
                if component['y'] == 0:
                    # Use a small non-zero value to avoid exact zero issues
                    component['y'] = 1e-10
                mirrored_component = component.copy()
                mirrored_component['y'] = -component['y']
                components[component_type].append(mirrored_component)

    return components


def generate_circular_vertices(center, radius, num_segments):
    angle_step = 2 * np.pi / num_segments
    vertices = []
    for i in range(num_segments):
        angle = i * angle_step
        x = center[0]
        y = center[1] + radius * np.cos(angle)
        z = center[2] + radius * np.sin(angle)
        vertices.append([x, y, z])
    return vertices


def generate_fuselage_vertices_and_faces(fuselage, num_segments=16):
    # Extract fuselage parameters
    ax, ay, az = fuselage['ax'], fuselage['ay'], fuselage['az']
    bx, by, bz = fuselage['bx'], fuselage['by'], fuselage['bz']
    width = fuselage['width']
    taper = fuselage['taper']
    midpoint = fuselage['midpoint']

    # Calculate positions
    front_center = np.array([ax, ay, az])
    back_center = np.array([bx, by, bz])

    # Calculate widths at front, midpoint, and back
    front_width = width * taper
    mid_width = width
    back_width = width * taper

    # Calculate circular cross-sections
    front_vertices = generate_circular_vertices(
        front_center, front_width / 2, num_segments)
    mid_center = front_center + (back_center - front_center) * midpoint
    mid_vertices = generate_circular_vertices(
        mid_center, mid_width / 2, num_segments)
    back_vertices = generate_circular_vertices(
        back_center, back_width / 2, num_segments)

    # Combine vertices
    vertices = np.array(front_vertices + mid_vertices + back_vertices)

    # Define faces by connecting corresponding vertices of front, middle, and
    # back circles
    faces = []
    num_front = len(front_vertices)
    num_mid = len(mid_vertices)

    # Create faces between front and middle
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        faces.append([i, next_i, num_front + next_i, num_front + i])

    # Create faces between middle and back
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        faces.append([num_front + i, num_front + next_i,
                     num_front + num_mid + next_i, num_front + num_mid + i])

    return vertices, faces


def generate_component_vertices_and_faces(component, component_type):
    # Extract wing parameters
    x, y, z = component['x'], component['y'], component['z']
    length = component['length']
    chord = component['chord']
    taper = component['taper']
    sweep = component['sweep']
    dihedral = component['dihedral']
    incidence = component['incidence']
    twist = component['twist']

    # Calculate the tip chord length
    tip_chord = chord * taper

    # Calculate root and tip positions
    root_center = np.array([x, y, z])

    # Horizontal offset due to sweep
    sweep_offset = length * np.sin(np.radians(sweep))
    span_offset = length * np.cos(np.radians(sweep))

    # Calculate tip center considering sweep
    tip_center = root_center + np.array([
        -sweep_offset,  # x direction (sweep effect) reversed
        0,              # y direction remains unchanged
        span_offset     # z direction remains unchanged
    ])

    # Define the thickness of the stabilizer using a typical
    # thickness-to-chord ratio
    root_thickness = chord * THICKNESS_TO_CHORD_RATIO
    tip_thickness = tip_chord * THICKNESS_TO_CHORD_RATIO

    # Generate vertices for root
    root_vertices = [
        root_center + np.array([-chord / 2, -root_thickness / 2, 0]),
        root_center + np.array([chord / 2, -root_thickness / 2, 0]),
        root_center + np.array([chord / 2, root_thickness / 2, 0]),
        root_center + np.array([-chord / 2, root_thickness / 2, 0])
    ]

    # Generate vertices for tip
    tip_vertices = [
        tip_center + np.array([-tip_chord / 2, -tip_thickness / 2, 0]),
        tip_center + np.array([tip_chord / 2, -tip_thickness / 2, 0]),
        tip_center + np.array([tip_chord / 2, tip_thickness / 2, 0]),
        tip_center + np.array([-tip_chord / 2, tip_thickness / 2, 0])
    ]

    # Mid-chord offset for twist application
    root_mid_chord = (root_vertices[0] + root_vertices[1]) / 2
    tip_mid_chord = (tip_vertices[0] + tip_vertices[1]) / 2

    # Calculate the average mid-chord position along the x-axis
    mid_chord_position = (root_mid_chord + tip_mid_chord) / 2
    mid_chord_position[1] = (root_center[1] + tip_center[1]) / 2

    # Apply twist to the tip vertices around the z-axis
    twist_radians = np.radians(-twist)  # Reverse the twist direction
    twist_matrix = np.array([
        [np.cos(twist_radians), -np.sin(twist_radians), 0],
        [np.sin(twist_radians), np.cos(twist_radians), 0],
        [0, 0, 1]
    ])

    # Translate tip vertices to align with mid-chord, apply twist, and
    # translate back
    tip_vertices_centered = [
        vertex - mid_chord_position for vertex in tip_vertices]
    twisted_tip_vertices_centered = [
        np.dot(twist_matrix, vertex) for vertex in tip_vertices_centered]
    twisted_tip_vertices = [
        vertex +
        mid_chord_position for vertex in twisted_tip_vertices_centered]

    # Combine root and twisted tip vertices
    vertices = np.array(root_vertices + twisted_tip_vertices)

    # Calculate the midpoint of the root chord, considering thickness
    root_midpoint = (
        root_vertices[0] + root_vertices[1] + root_vertices[2] + root_vertices[3]) / 4

    # Translate vertices to move the root midpoint to the origin
    translated_vertices = vertices - root_midpoint

    # Apply incidence rotation around the z-axis (left-right tilt) in the
    # reversed direction
    incidence_radians = np.radians(-incidence)
    incidence_matrix = np.array([
        [np.cos(incidence_radians), -np.sin(incidence_radians), 0],
        [np.sin(incidence_radians), np.cos(incidence_radians), 0],
        [0, 0, 1]
    ])

    rotated_vertices = np.dot(translated_vertices, incidence_matrix.T)

    # Apply dihedral tilt around the x-axis (reversed direction: positive angle tilts left)
    # Reverse the dihedral direction
    dihedral_radians = np.radians(90 - dihedral)
    dihedral_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(dihedral_radians), np.sin(dihedral_radians)],
        [0, -np.sin(dihedral_radians), np.cos(dihedral_radians)]
    ])

    rotated_vertices = np.dot(rotated_vertices, dihedral_matrix.T)

    # Translate vertices back to their original position
    final_vertices = rotated_vertices + root_midpoint

    # Define faces
    faces = [
        [0, 1, 5, 4],  # Bottom face
        [2, 3, 7, 6],  # Top face
        [0, 1, 2, 3],  # Front face
        [4, 5, 6, 7],  # Back face
        [0, 4, 7, 3],  # Side face
        [1, 5, 6, 2]   # Side face
    ]

    # If the component is on the negative side of the y-axis, mirror it and
    # offset it to the correct side
    if y < 0:
        final_vertices[:, 1] = -final_vertices[:, 1] - 2 * abs(y)

    return final_vertices, faces


def create_sphere_polydata(
        radius,
        center,
        theta_resolution=8,
        phi_resolution=8):
    return pv.Sphere(
        radius=radius,
        center=center,
        theta_resolution=theta_resolution,
        phi_resolution=phi_resolution)


def get_center_of_mass(vertices):
    return np.mean(vertices, axis=0)


def add_label_at_offset(
        plotter,
        vertices,
        label,
        label_id,
        offset=(
            0,
            0,
            1),
        show_labels=False):
    if show_labels:
        center = get_center_of_mass(vertices)
        offset_position = center + np.array(offset)
        plotter.add_point_labels(
            [offset_position], [
                f'{label} {label_id}'], font_size=POINT_LABEL_FONT_SIZE, bold=True)


def add_weight_label_at_offset(
        plotter,
        center,
        label,
        label_id,
        offset=(
            0,
            0,
            0.5),
    show_labels=False,
        weights=False):
    if show_labels or weights:
        offset_position = np.array(center) + np.array(offset)
        plotter.add_point_labels(
            [offset_position], [
                f'{label} {label_id}'], font_size=POINT_LABEL_FONT_SIZE, bold=True)


def add_mesh_with_options(plotter, mesh, color, label, alpha):
    plotter.add_mesh(mesh, color=color, show_edges=False, smooth_shading=True,
                     opacity=alpha, label=label)


def singularize_label(label):
    # Basic singularization logic for common cases
    if label.endswith('s'):
        return label[:-1]  # Remove the trailing 's' for basic plural forms
    return label


def process_component_or_weight(
        plotter,
        components,
        component_type,
        color_key,
        ids,
        COLORS,
        alpha,
        show_labels=False,
        weights=False):
    singular_type = singularize_label(component_type)

    for component in components[component_type]:
        if weights:
            center = (component['x'], component['y'], component['z'])
            sphere_polydata = create_sphere_polydata(
                radius=WEIGHT_SPHERE_RADIUS, center=center)

            # Retrieve 'mass-prop' for weights and 'mass' for ballasts
            mass_prop = component.get(
                'mass-prop', 'N/A') if component_type == 'weights' else 'N/A'
            mass = component.get(
                'mass', 'N/A') if component_type == 'ballasts' else component.get('capacity', 'N/A')

            # Determine which mass to use based on component type
            if component_type == 'weights' and mass_prop != 'N/A':
                label_text = f'{singular_type} {ids[color_key]}: {mass_prop}'
            else:
                label_text = f'{singular_type} {ids[color_key]}: {mass} lbs'

            plotter.add_mesh(
                sphere_polydata,
                color=COLORS[color_key],
                show_edges=False,
                smooth_shading=True,
                label=label_text)
            add_weight_label_at_offset(
                plotter,
                center,
                singular_type,
                ids[color_key],
                show_labels=show_labels,
                weights=weights)
        else:
            if component_type == 'fuselages':
                vertices, faces = generate_fuselage_vertices_and_faces(
                    component)
            else:
                vertices, faces = generate_component_vertices_and_faces(
                    component, component_type)
            faces = np.hstack([[len(face)] + face for face in faces])
            mesh = pv.PolyData(vertices, faces)
            smoothed_mesh = mesh.smooth(n_iter=20, relaxation_factor=0.01)
            add_mesh_with_options(
                plotter,
                smoothed_mesh,
                COLORS[color_key],
                f'{singular_type} {
                    ids[color_key]}',
                alpha)
            if show_labels:
                add_label_at_offset(
                    plotter,
                    vertices,
                    singular_type,
                    ids[color_key],
                    show_labels=show_labels)

        ids[color_key] += 1


def visualize_with_pyvista(
        components,
        xml_file,
        show_labels=False,
        transparency=False,
        weights=False,
        background_image=None):
    plotter = pv.Plotter()

    # Set the title
    plotter.title = f"YASimViz - {xml_file}"

    # Handle background image
    if background_image and os.path.isfile(background_image):
        try:
            plotter.add_background_image(
                background_image,
                scale=1.0,
                auto_resize=True,
                as_global=False)
        except FileNotFoundError:
            print(
                f"Warning: The background image file '{background_image}' could not be loaded. It will be ignored.")
    else:
        if background_image:
            print(
                f"Warning: The background image file '{background_image}' does not exist. It will be ignored.")
        plotter.set_background(COLORS['background'])

    alpha = 0.5 if transparency or weights else 1.0

    ids = {key: 0 for key in COLORS.keys()}

    if weights:
        # Process weight components first
        for weight_type in ['tanks', 'ballasts', 'weights']:
            process_component_or_weight(
                plotter,
                components,
                weight_type,
                weight_type[:-1],
                ids,
                COLORS,
                alpha,
                show_labels=show_labels,
                weights=True
            )

        # Add legend after all components are processed
        plotter.add_legend()

    # Process non-weight components
    for component_type in ['fuselages', 'wings', 'hstabs', 'vstabs', 'mstabs']:
        process_component_or_weight(
            plotter,
            components,
            component_type,
            component_type[:-1],
            ids,
            COLORS,
            alpha,
            show_labels=show_labels,
            weights=False
        )

    if not weights:
        # Add legend after all components are processed
        plotter.add_legend()

    # Initialize the camera mode text
    camera_mode_text = plotter.add_text(
        'Camera: Perspective',
        position='lower_left',
        font_size=CAMERAMODE_FONT_SIZE,
        color=COLORS['cameramode'])

    # Function to reset the view to head-on from the positive x-axis
    def reset_view():
        plotter.camera_position = [(10, 0, 0), (0, 0, 0), (0, 0, 1)]
        plotter.reset_camera()

    # Functions to rotate the view
    def rotate_x():
        plotter.camera.roll += 45
        plotter.render()

    def rotate_y():
        plotter.camera.azimuth += 45
        plotter.render()

    def rotate_z():
        plotter.camera.elevation += 45
        plotter.render()

    # Functions to fine-tune the zoom
    def zoom_in():
        plotter.camera.zoom(1.01)
        plotter.render()

    def zoom_out():
        plotter.camera.zoom(0.99)
        plotter.render()

    # Function to cycle transparency
    def cycle_transparency():
        nonlocal alpha
        # Increment by 0.1 and round to avoid floating point issues
        alpha = round(alpha + 0.1, 1)
        if alpha > 1.0:
            alpha = 0.1
        # Update transparency for all components
        for actor in plotter.actors.values():
            actor.GetProperty().SetOpacity(alpha)
        plotter.render()

    # Function to toggle measurement widget
    measurement_state = [False]

    def toggle_measurement():
        if measurement_state[0]:
            # Disable measurement widget
            plotter.clear_measure_widgets()
        else:
            # Enable measurement widget
            plotter.add_measurement_widget(color='green')
        plotter.render()
        measurement_state[0] = not measurement_state[0]

    # Function to toggle between orthographic and perspective camera views and
    # update text
    def toggle_projection():
        if plotter.camera.GetParallelProjection():
            plotter.camera.ParallelProjectionOff()
            camera_mode_text.SetText(0, 'Camera: Perspective')
        else:
            plotter.camera.ParallelProjectionOn()
            camera_mode_text.SetText(0, 'Camera: Orthographic')
        plotter.render()

    # Bind the reset view function to the 'c' key
    plotter.add_key_event('c', reset_view)
    plotter.add_key_event('x', rotate_x)
    plotter.add_key_event('y', rotate_y)
    plotter.add_key_event('z', rotate_z)
    plotter.add_key_event('Up', zoom_in)
    plotter.add_key_event('Down', zoom_out)
    plotter.add_key_event('t', cycle_transparency)
    plotter.add_key_event('m', toggle_measurement)
    plotter.add_key_event('o', toggle_projection)

    # Show the plot
    plotter.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YASim XML data with PyVista.")
    parser.add_argument('xml_file', type=str,
                        help="Path to the YASim XML file.")
    parser.add_argument('-l', '--labels', action='store_true',
                        help="Display labels for components.")
    parser.add_argument('-t', '--transparency', action='store_true',
                        help="Make objects semi-transparent.")
    parser.add_argument('-w', '--weights', action='store_true',
                        help='Display weight elements.')
    parser.add_argument('-b', '--background-image', type=str,
                        help='Path to a background image file.')
    args = parser.parse_args()

    # Check if XML file exists
    if not os.path.isfile(args.xml_file):
        print(f"Error: The XML file '{args.xml_file}' does not exist.")
        return

    try:
        components = parse_yasim_file(args.xml_file)
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        return

    # Check if background image file exists (if provided)
    if args.background_image and not os.path.isfile(args.background_image):
        print(
            f"Warning: The background image file '{
                args.background_image}' does not exist. It will be ignored.")
        background_image = None
    else:
        background_image = args.background_image

    visualize_with_pyvista(
        components,
        xml_file=args.xml_file,
        show_labels=args.labels,
        transparency=args.transparency,
        weights=args.weights,
        background_image=background_image
    )


if __name__ == '__main__':
    main()
