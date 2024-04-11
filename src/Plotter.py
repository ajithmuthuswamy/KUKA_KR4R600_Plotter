import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from tkinter.filedialog import askopenfilename
from tkinter import Tk, simpledialog, messagebox
    
# FIND AND STORE CONTOUR POINTS
# Open file dialog box to select the image
Tk().withdraw()  # Hide the root window
initial_dir = "/home/ak/Documents/KUKA_KR4R600_Plotter/data"

# Define allowed image file extensions
image_extensions = (".png", ".jpg", ".jpeg")
image_file = askopenfilename(initialdir = initial_dir, filetypes=[("Image Files", image_extensions)])

# Read the image
img = cv2.imread(image_file)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove noise
blur = cv2.GaussianBlur(gray, (3, 3),0)

# Detect edges using Canny edge detector
def auto_canny(image, sigma=0.33):
     # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
edges = auto_canny(blur)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # Parameters to keep in mind: RETR_EXTERNAL (external contours), CHAIN_APPROX_SIMPLE (approx), CHAIN_APPROX_NONE

# SIMPLIFY CONTOUR POINTS - remove unnecessary contours
# Ask for epsilon value
eps = simpledialog.askfloat("Epsilon", "Enter the epsilon value (0.001 - 0.01):\n"
                                      "Smaller value generates more contour points",
                           minvalue=0.001, maxvalue=0.01)


epsilon = eps * cv2.arcLength(contours[0], True)  # Set the epsilon value as desired - smaller value generates more contours
simplified_contour = cv2.approxPolyDP(contours[0], epsilon, True)

# Initialize empty list to store coordinates
coordinates = []
sub_coords = []
for point in simplified_contour[:, 0]:
    x, y = point
    sub_coords.append((x, y))

# Append sub_coords to coordinates after the loop
coordinates.extend(sub_coords)
total_points = 0
for contour in contours:
    num_points = len(contour)
    total_points += num_points

# TRANSFORM PIXEL COORDINATES TO MM
# Set up pixel resolution per inch
dpi = 300

# Constants for A4 page dimensions in mm
A4_WIDTH_MM = 297  # 25mm for the margin
A4_HEIGHT_MM = 210  # 25mm for the margin

# Calculate the size of the image in mm
width_mm = img.shape[1] * 25.4 / dpi
height_mm = img.shape[0] * 25.4 / dpi
messagebox.showinfo("Image Size in MM", f"Width: {round(width_mm,2)} mm\nHeight: {round(height_mm,2)} mm")
# Convert the coordinates to mm
converted_coordinates = []
while True:
    scaling_factor = simpledialog.askfloat("Scaling Factor", "Enter the scaling factor with one decimal place:")

    converted_coordinates = []  # Reset converted coordinates
    
    for coord in coordinates:
        x_mm = ((coord[0] / img.shape[1]) * width_mm) * scaling_factor  # Convert x-coordinate to mm with scaling
        y_mm = (((img.shape[0] - coord[1]) / img.shape[0]) * height_mm) * scaling_factor  # Convert y-coordinate to mm with scaling
        z_mm = 0.0
        converted_coordinates.append((x_mm, y_mm, z_mm))

    # Calculate the estimated width and height based on the difference between maximum and minimum x and y values
    min_x = min(converted_coordinates, key=lambda c: c[0])[0]
    max_x = max(converted_coordinates, key=lambda c: c[0])[0]
    min_y = min(converted_coordinates, key=lambda c: c[1])[1]
    max_y = max(converted_coordinates, key=lambda c: c[1])[1]
    width_img = max_x - min_x
    height_img = max_y - min_y
    
    # Check if the estimated dimensions exceed A4 page size
    if width_img > A4_WIDTH_MM or height_img > A4_HEIGHT_MM:
        messagebox.showerror("Error", "The image is too big for an A4 page. Please enter a new scaling factor.")
    elif scaling_factor == 0 or scaling_factor == None:
        messagebox.showerror("Error", "Please enter a valid scaling factor greater than 0.")
    else:
        # Display the estimated dimensions in a dialog box
        message = f"Estimated Size:\nWidth: {round(width_img,2)} mm\nHeight: {round(height_img,2)} mm"
        messagebox.showinfo("Estimated Size", message)
        break
# Calculate the center of the A4 paper
center_x = A4_WIDTH_MM / 2
center_y = A4_HEIGHT_MM / 2

# Calculate the centroid of the converted coordinates
centroid_x = sum(coord[0] for coord in converted_coordinates) / len(converted_coordinates)
centroid_y = sum(coord[1] for coord in converted_coordinates) / len(converted_coordinates)

# Calculate the offset to center the drawing
offset_x = center_x - centroid_x
offset_y = center_y - centroid_y

# Apply the offset to the converted coordinates
centered_coordinates = [(coord[0] + offset_x, coord[1] + offset_y, coord[2]) for coord in converted_coordinates] 
plot_coordinates = [(coord[0] + offset_x, coord[1] + offset_y) for coord in converted_coordinates]

# Lift, lower the pen, and close the contour
centered_coordinates.insert(0, (centered_coordinates[0][0], centered_coordinates[0][1], 50)) # Hover over the first point
centered_coordinates.append((centered_coordinates[0][0], centered_coordinates[0][1], 0)) # Close the contour
centered_coordinates.append(centered_coordinates[0]) # Hover over the last point

# CREATE THE .DAT AND .SRC FILES
# Set the file name, base number, and tool number
directory = "/home/ak/Documents/KUKA_KR4R600_Plotter/results"

# Ask for file name
fname = None
while not fname or ' ' in fname:
    fname = simpledialog.askstring("File Name", "Enter the file name (without spaces):")

    if not fname or ' ' in fname:
        messagebox.showerror("Invalid File Name", "Please enter a valid file name without spaces.")

file_path = os.path.join(directory, fname)

# Generate .dat file
with open(file_path + ".dat", "w") as f:
    f.write("DEFDAT " + fname + "\n")
    f.write("   EXT BAS(BAS_COMMAND :IN, REAL :IN)\n")
    f.write("   DECL PDAT PDEFAULT={APO_MODE #CDIS,APO_DIST 100.0,VEL 100.0,ACC 100.0,GEAR_JERK 100.0,EXAX_IGN 0}\n")
    for i, coord in enumerate(centered_coordinates):
        # f.write("   DECL E6POS XP{} = {{X {:.2f}, Y {:.2f}, Z {:.2f}, A 140.40, B 89.07, C -130.82, S 2, T 34, E1 0.0, E2 0.0, E3 0.0, E4 0.0, E5 0.0, E6 0.0 }}\n".format(i+10, coord[0], coord[1], coord[2]))
        f.write("   DECL E6POS XP{} = {{X {:.2f}, Y {:.2f}, Z {:.2f}, A 140.40, B 90, C -130.82, S 2, T 34}}\n".format(i+10, coord[0], coord[1], coord[2]))
        # f.write('   DECL FDAT FP{} = {{ BASE_NO {}, TOOL_NO {}, IPO_FRAME #BASE, POINT2[] " " }}\n'.format(i+10, BASE, TOOL))
    f.write("ENDDAT")
f.close()

# Generate the .src file
with open(file_path + ".src", "w") as f:
    f.write("DEF " + fname + "()\n")
    f.write("\t   GLOBAL INTERRUPT DECL 3 WHEN $STOPMESS == TRUE DO IR_STOPM()\n")
    f.write("\t   INTERRUPT ON 3\n")
    f.write("\t   BAS(#INITMOV, 0)\n")
    f.write("   ; Movement list to pick up the pen\n")
    f.write("   PenPickUp()\n")
    f.write("   ; Set up pen tool data and white paper as base\n")
    f.write("   $TOOL = {X 94.08, Y -1.55, Z 212.98, A -1.42, B -59.72, C 0.0 }\n")
    f.write("   $BASE = {X 244.87, Y -26.5, Z 177.45, A -90.0, B 0.0, C 0.0 }\n")
    # Loop through the coordinates and write the LIN commands to the file
    for i, coord in enumerate(centered_coordinates):
        # Write the LIN command to the file
        f.write("   SLIN XP{}\n".format(i + 10))
    f.write("   ; Movement list to return the pen\n")
    f.write("   Return1()\n")
    f.write("END")
# Close the file
f.close()

# DISPLAY THE CONTOUR POINTS
# Create blank white image
white_img = np.zeros_like(img) + 255

# Initialize subplot with 1 row and 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot the original image
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Image")

# Plot the raw contours
cv2.drawContours(white_img, contours, -1, (0, 255, 0), 2)
axs[1].imshow(cv2.cvtColor(white_img, cv2.COLOR_BGR2RGB))
axs[1].set_title("Raw Contours")

# Plot the simplified contours
axs[2].plot([p[0] for p in coordinates], [p[1] for p in coordinates], 'ro', markersize=2)

# Add numbers to each point
for i, point in enumerate(sub_coords):
    axs[2].text(point[0], point[1], str(i+1), ha='center', va='bottom')
axs[2].invert_yaxis()  # invert the y-axis
axs[2].set_title("Simplified Contours")

# Adjust the spacing between subplots
plt.savefig("/home/ak/Documents/KUKA_KR4R600_Plotter/results/plot-"+str(fname)+".png")
plt.tight_layout()

# Create the plot with A4 dimensions
fig2, ax2 = plt.subplots(figsize=(8.27, 11.69))  # A4 size in inches

# Create a gridspec layout
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

# Set the aspect ratio to match A4 dimensions for the left subplot
ax2.set_aspect('equal', adjustable='box')

# Plot all the coordinates with red circles
for coord in plot_coordinates:
    ax2.scatter(coord[0], coord[1], color='red')
# Connect the points with lines
x_values, y_values = zip(*plot_coordinates)
min_x = min(x_values)
max_x = max(x_values)
min_y = min(y_values)
max_y = max(y_values)

width = round(max_x - min_x,2)
height = round(max_y - min_y,2)

ax2.plot(x_values, y_values, color='blue')
ax2.plot([x_values[-1], x_values[0]], [y_values[-1], y_values[0]], color='blue')  # Connect last and first points

# Draw the bounding box
bbox = plt.Rectangle((min_x, min_y), width, height, fill=False, edgecolor='green', linewidth=0.5, linestyle='dotted')
plt.gca().add_patch(bbox)

# Set the x and y limits to match A4 dimensions
ax2.set_xlim(0, 297)
ax2.set_ylim(0, 210)

# Add labels to the points
for i, coord in enumerate(plot_coordinates):
    ax2.text(coord[0], coord[1], str(i+1), ha='center', va='bottom')

# Set the axis labels and title
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_title(f'$\mathbf{{Drawing\ Visualization\ on\ A4\ Paper}}$')

# Add text annotations for the information
info_text = f"$\mathbf{{Number\ of\ contours:}}$ {len(contours)}\n" \
            f"$\mathbf{{Number\ of\ raw\ contour\ points:}}$ {total_points}\n" \
            f"$\mathbf{{Epsilon:}}$ {epsilon:.2f}\n" \
            f"$\mathbf{{Number\ of\ simplified\ contour\ points:}}$ {len(coordinates)}\n" \
            f"$\mathbf{{Pixel\ size:}}$ {img.shape[1]} x {img.shape[0]}\n" \
            f"$\mathbf{{DPI:}}$ {dpi}\n" \
            f"$\mathbf{{Scaling\ factor:}}$ {scaling_factor}\n" \
            f"$\mathbf{{Drawing\ size:}}$ {width} x {height} mm\n" \
            f"$\mathbf{{File\ name:}}$ {fname}"

# Add the text annotation on the right side
ax_info = fig2.add_subplot(gs[0, 1])
ax_info.axis('off')
ax_info.text(0, 0.5, info_text, ha='left', va='center', fontsize=10, family='monospace', bbox=dict(facecolor='white', edgecolor='black'))

# Show the plots
plt.savefig("/home/ak/Documents/KUKA_KR4R600_Plotter/results/res-"+str(fname)+".png")
plt.show()
# Close the figures
plt.close('all')



# WORK IN PROGRESS!

contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
# Simplify contours - remove unnecessary contours points
simplified_contours = []
epsilon = 0.003  # Set the epsilon value as desired - smaller value generates more contours

for contour in contours:
    simplified_contour = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
    simplified_contours.append(simplified_contour)


for contour in simplified_contours:
    cv2.drawContours(img, contour, -1, (0, 255, 0), 2)
num_points = [contour.shape[0] for contour in simplified_contours]
print(sum(num_points))

filtered_contours = []
area_threshold = 3000  # Adjust this threshold as needed - issue resides in the fact that this was found by trial and error

for contour in simplified_contours:
    area = cv2.contourArea(contour)
    is_overlapping = False

    for filtered_contour in filtered_contours:
        filtered_area = cv2.contourArea(filtered_contour)
        if abs(area - filtered_area) < area_threshold:
            is_overlapping = True
            break

    if not is_overlapping:
        filtered_contours.append(contour)

num_points = [contour.shape[0] for contour in filtered_contours]
print(sum(num_points))

# Create a blank image of the same size as the original image
image_lines = np.zeros_like(img)

# Create figure 3 for the plot
plt.figure(3)

# Define colors for each contour
colors = ['red', 'green']

# Initialize a counter for the total number of plotted points
total_points = 0

# Plot each contour separately with a different color
for i, contour in enumerate(filtered_contours):
    x, y = contour[:, 0, 0], contour[:, 0, 1]
    plt.scatter(x, y, color=colors[i % len(colors)], marker='o')

    # Add the number offset by the total number of plotted points
    for j, (x_val, y_val) in enumerate(zip(x, y)):
        plt.text(x_val, y_val, str(j + 1 + total_points), ha='center', va='bottom')

    # Update the total number of plotted points
    total_points += len(x)

# Set the aspect ratio to 'equal'
plt.gca().set_aspect('equal')

# Add a legend
inner_patch = mpatches.Patch(color='green', label='Outer Contours')
outer_patch = mpatches.Patch(color='red', label='Inner Contours')
plt.legend(handles=[inner_patch, outer_patch])
# Set the axis labels and title
ax2.set_title(f'$\mathbf{{Image\ with\ inner\ and\ outer\ contours}}$')

# Display the plot
plt.axis('image')
plt.gca().invert_yaxis()
plt.show()

# Pause to keep the plot visible for a certain duration (e.g., 5 seconds)
plt.pause(1)

# Close the figures
plt.close('all')

