import tkinter as tk
import matplotlib
matplotlib.use('Agg')
import torch
from MLP import MLP
import pandas as pd

# Create an instance of the MLP model
model = MLP(input_size=10, hidden_size=10, output_size=1)

# Load the pre-trained weights from file
model_weights_path = "model_weights/sparkling-rain-11957.pth"
model.load_state_dict(torch.load(model_weights_path))
model.eval()

# Read the min and max values from the saved file
min_max_df = pd.read_excel("min_max_values.xlsx")

# Create a dictionary to store the min and max values for each column
min_max_dict = {}
for i, row in min_max_df.iterrows():
    min_max_dict[row['column']] = (row['min_value'], row['max_value'])

# Create a new GUI window
root = tk.Tk()
root.title("Engineering School Metrics")

# Set the window background color to Rowan University's color scheme
root.configure(bg='#57150B')

# Add image to the GUI window
logo = tk.PhotoImage(file="USNewsLogo.png")
logo_label = tk.Label(root, image=logo)
logo_label.grid(row=0, column=0, columnspan=2)

# Default values
defaults = [2, 2.6, 160.65, 41.6, 1, 4.07, 8.9, 369.0, 8, 424]

# Create labels and entry fields for each parameter
peer_label = tk.Label(root, text="Peer assessment score (5.0=highest):", bg='#FFCC00')
peer_entry = tk.Entry(root)
peer_entry.insert(0, defaults[0]) # Add default value to entry field
recruiter_label = tk.Label(root, text="Recruiter assessment score (5.0=highest):", bg='#FFCC00')
recruiter_entry = tk.Entry(root)
recruiter_entry.insert(0, defaults[1]) # Add default value to entry field
gre_label = tk.Label(root, text="2021 average quantitative GRE score:", bg='#FFCC00')
gre_entry = tk.Entry(root)
gre_entry.insert(0, defaults[2]) # Add default value to entry field
acceptance_label = tk.Label(root, text="2021 acceptance rate:", bg='#FFCC00')
acceptance_entry = tk.Entry(root)
acceptance_entry.insert(0, defaults[3]) # Add default value to entry field
phd_faculty_label = tk.Label(root, text="2021 Ph.D. students/faculty:", bg='#FFCC00')
phd_faculty_entry = tk.Entry(root)
phd_faculty_entry.insert(0, defaults[4]) # Add default value to entry field
nae_label = tk.Label(root, text="2021 faculty membership in National Academy of Engineering:", bg='#FFCC00')
nae_entry = tk.Entry(root)
nae_entry.insert(0, defaults[5]) # Add default value to entry field
research_label = tk.Label(root, text="2021 engineering school research expenditures (in millions):", bg='#FFCC00')
research_entry = tk.Entry(root)
research_entry.insert(0, defaults[6]) # Add default value to entry field
research_faculty_label = tk.Label(root, text="2021 research expenditures per faculty member (in thousands):", bg='#FFCC00')
research_faculty_entry = tk.Entry(root)
research_faculty_entry.insert(0, defaults[7]) # Add default value to entry field
phds_label = tk.Label(root, text="Ph.D.s granted in 2020-2021:", bg='#FFCC00')
phds_entry = tk.Entry(root)
phds_entry.insert(0, defaults[8]) # Add default value to entry field
enrollment_label = tk.Label(root, text="2021 total graduate engineering enrollment:", bg='#FFCC00')
enrollment_entry = tk.Entry(root)
enrollment_entry.insert(0, defaults[9]) # Add default value to entry field

# Place the labels and entry fields in the window
peer_label.grid(row=1, column=0, padx=5, pady=5, sticky="E")
peer_entry.grid(row=1, column=1, padx=5, pady=5)
recruiter_label.grid(row=2, column=0, padx=5, pady=5, sticky="E")
recruiter_entry.grid(row=2, column=1, padx=5, pady=5)
gre_label.grid(row=3, column=0, padx=5, pady=5, sticky="E")
gre_entry.grid(row=3, column=1, padx=5, pady=5)
acceptance_label.grid(row=4, column=0, padx=5, pady=5, sticky="E")
acceptance_entry.grid(row=4, column=1, padx=5, pady=5)
phd_faculty_label.grid(row=5, column=0, padx=5, pady=5, sticky="E")
phd_faculty_entry.grid(row=5, column=1, padx=5, pady=5)
nae_label.grid(row=6, column=0, padx=5, pady=5, sticky="E")
nae_entry.grid(row=6, column=1, padx=5, pady=5)
research_label.grid(row=7, column=0, padx=5, pady=5, sticky="E")
research_entry.grid(row=7, column=1, padx=5, pady=5)
research_faculty_label.grid(row=8, column=0, padx=5, pady=5, sticky="E")
research_faculty_entry.grid(row=8, column=1, padx=5, pady=5)
phds_label.grid(row=9, column=0, padx=5, pady=5, sticky="E")
phds_entry.grid(row=9, column=1, padx=5, pady=5)
enrollment_label.grid(row=10, column=0, padx=5, pady=5, sticky="E")
enrollment_entry.grid(row=10, column=1, padx=5, pady=5)

# Define a function to normalize the user inputs based on the min and max values
def normalize_inputs(inputs):
    normalized_inputs = []
    for i in range(len(inputs)):
        col_name = min_max_df.iloc[i,0]  # get the name of the column to normalize
        min_val, max_val = min_max_dict[col_name]  # get the min and max values for the column
        input_val = inputs[i]  # get the user input for the column
        normalized_val = (input_val - min_val) / (max_val - min_val)  # normalize the input
        normalized_inputs.append(normalized_val)
    return normalized_inputs

# Function to retrieve values entered by the user and perform calculations
def calculate_metrics():
    # Retrieve values from entry fields
    peer_score = float(peer_entry.get())
    recruiter_score = float(recruiter_entry.get())
    gre_score = float(gre_entry.get())
    acceptance_rate = float(acceptance_entry.get())
    phd_faculty_ratio = float(phd_faculty_entry.get())
    nae_membership = float(nae_entry.get())
    research_expenditures = float(research_entry.get())
    research_faculty_expenditures = float(research_faculty_entry.get())
    phd_count = float(phds_entry.get())
    enrollment = float(enrollment_entry.get())

    # Normalize the user inputs based on the min and max values
    normalized_inputs = normalize_inputs([peer_score, recruiter_score, gre_score, acceptance_rate, phd_faculty_ratio,
                                           nae_membership, research_expenditures, research_faculty_expenditures,
                                           phd_count, enrollment])

    normalized_inputs = torch.tensor(normalized_inputs).unsqueeze(0)
   
    # Calculate metrics
    score = model(normalized_inputs)
    score = float(score.item())
    score_label.config(text=f"Score: {score:.2f}")  # Update the score label text

# Create a label to display the score
score_label = tk.Label(root, text="Score: ", font=("Arial", 16), bg='#FFCC00')
score_label.grid(row=11, column=0, columnspan=2, pady=10)

# Create a "Calculate Score" button
calculate_button = tk.Button(root, text="Calculate Score", command=calculate_metrics, bg="#FFCC00")
calculate_button.grid(row=12, column=0, columnspan=2, pady=10)

# Run the GUI window
root.mainloop()