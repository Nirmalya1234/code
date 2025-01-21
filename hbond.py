import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

# Load the trajectory and topology
u = mda.Universe("nvt_10fs.gro", "nvt_10fs_10ps.trr")

# Define the selection of methyl hydrogens, donor atoms, and water oxygens
dmso_hydrogens = u.select_atoms("name HD* and resid 2")  # Adjust if needed
dmso_carbons = u.select_atoms("name CD* and resid 2")    # Donors for methyl hydrogens
water_oxygens = u.select_atoms("name OW and resname SOL")  # Adjust resname if needed

print(f"DMSO Hydrogens: {len(dmso_hydrogens)}")
print(f"DMSO Carbons: {len(dmso_carbons)}")
print(f"Water Oxygens: {len(water_oxygens)}")

# Function to calculate H-bonds for the entire trajectory
def calculate_hbonds():
    hbond_counts = []
    for ts in u.trajectory:
        unique_hbonds = 0
        frame_hbonds = []
        #used_oxygens = set()  # Track oxygen atoms already forming H-bonds

        for h, d in zip(dmso_hydrogens.positions, dmso_carbons.positions):
            #for o_idx, o in enumerate(water_oxygens.positions):
             #   if o_idx in used_oxygens:
              #      continue  # Skip if this oxygen is already forming an H-bond
            for o in water_oxygens.positions:
                # Calculate distance
                dist = np.linalg.norm(d - o)
                if dist <= 3.8:  # Distance cutoff
                    # Calculate angle (D-H···O)
                    dh_vector = h - d
                    ho_vector = o - h
                    cos_angle = np.dot(dh_vector, ho_vector) / (np.linalg.norm(dh_vector) * np.linalg.norm(ho_vector))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp for numerical stability
                    angle = np.degrees(np.arccos(cos_angle))

                    if 180.0 >= angle >= 120.0:  # Angle cutoff
                        # Lennard-Jones potential for interaction energy
                        epsilon = 0.2923  # Example value
                        sigma = 0.2832    # Example value
                        energy = 4 * epsilon * ((sigma / dist) ** 12 - (sigma / dist) ** 6)

                        if energy >= -1.0:  # Energy cutoff
                            unique_hbonds += 1
                            #used_oxygens.add(o_idx)  # Mark this oxygen as used
                            if unique_hbonds == 6:
                                break

                              # allow one H-bond per hydrogen

        hbond_counts.append(unique_hbonds)
    return hbond_counts

# Main code to calculate and analyze H-bonds
def main():
    hbonds_per_frame = calculate_hbonds()

    # Count occurrences of 0 to 6 H-bonds
    hbond_bins = list(range(7))  # Groups: 0 to 6 H-bonds
    population_counts = [0] * 7
    for count in hbonds_per_frame:
        if 0 <= count <= 6:
            population_counts[count] += 1

    # Calculate percentages
    total_frames = len(hbonds_per_frame)
    percentages = [100.0 * count / total_frames for count in population_counts]

    # Print results
    print(f"Total Frames: {total_frames}")
    for i, (count, percentage) in enumerate(zip(population_counts, percentages)):
        print(f"{i} H-bonds: {count} frames ({percentage:.2f}%)")

    # Plot population percentages for 0-6 H-bonds
    plt.figure(figsize=(8, 5))
    plt.bar(hbond_bins, percentages, color='skyblue', edgecolor='black', width=0.6)
    plt.xlabel("Number of H-bonds", fontsize=14)
    plt.ylabel("Population Percentage (%)", fontsize=14)
    plt.title("Distribution of H-bonds in Trajectory", fontsize=16)
    plt.xticks(hbond_bins)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("hbond_distribution_hf_6-311g_2.png")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
