import tkinter as tk
import matplotlib
matplotlib.use('Agg')

# Create a new GUI window
root = tk.Tk()
root.title("Engineering School Metrics")

# Create labels and entry fields for each parameter
peer_label = tk.Label(root, text="Peer assessment score (5.0=highest):")
peer_entry = tk.Entry(root)
recruiter_label = tk.Label(root, text="Recruiter assessment score (5.0=highest):")
recruiter_entry = tk.Entry(root)
gre_label = tk.Label(root, text="2021 average quantitative GRE score:")
gre_entry = tk.Entry(root)
acceptance_label = tk.Label(root, text="2021 acceptance rate:")
acceptance_entry = tk.Entry(root)
phd_faculty_label = tk.Label(root, text="2021 Ph.D. students/faculty:")
phd_faculty_entry = tk.Entry(root)
nae_label = tk.Label(root, text="2021 faculty membership in National Academy of Engineering:")
nae_entry = tk.Entry(root)
research_label = tk.Label(root, text="2021 engineering school research expenditures (in millions):")
research_entry = tk.Entry(root)
research_faculty_label = tk.Label(root, text="2021 research expenditures per faculty member (in thousands):")
research_faculty_entry = tk.Entry(root)
phds_label = tk.Label(root, text="Ph.D.s granted in 2020-2021:")
phds_entry = tk.Entry(root)
enrollment_label = tk.Label(root, text="2021 total graduate engineering enrollment:")
enrollment_entry = tk.Entry(root)

# Place the labels and entry fields in the window
peer_label.grid(row=0, column=0)
peer_entry.grid(row=0, column=1)
recruiter_label.grid(row=1, column=0)
recruiter_entry.grid(row=1, column=1)
gre_label.grid(row=2, column=0)
gre_entry.grid(row=2, column=1)
acceptance_label.grid(row=3, column=0)
acceptance_entry.grid(row=3, column=1)
phd_faculty_label.grid(row=4, column=0)
phd_faculty_entry.grid(row=4, column=1)
nae_label.grid(row=5, column=0)
nae_entry.grid(row=5, column=1)
research_label.grid(row=6, column=0)
research_entry.grid(row=6, column=1)
research_faculty_label.grid(row=7, column=0)
research_faculty_entry.grid(row=7, column=1)
phds_label.grid(row=8, column=0)
phds_entry.grid(row=8, column=1)
enrollment_label.grid(row=9, column=0)
enrollment_entry.grid(row=9, column=1)

# Function to retrieve values entered by the user and perform calculations
def calculate_metrics():
    # Retrieve values from entry fields
    peer_score = float(peer_entry.get())
    recruiter_score = float(recruiter_entry.get())
    gre_score = float(gre_entry.get())
    acceptance_rate = float(acceptance_entry.get())
    phd_faculty_ratio = float(phd_faculty_entry.get())
    nae_membership = int(nae_entry.get())
    research_expenditures = float(research_entry.get())
    research_faculty_expenditures = float(research_faculty_entry.get())
    phd_count = int(phds_entry.get())
    enrollment = int(enrollment_entry.get())

    # Calculate metrics
    score = (peer_score + recruiter_score + gre_score) / 15.
