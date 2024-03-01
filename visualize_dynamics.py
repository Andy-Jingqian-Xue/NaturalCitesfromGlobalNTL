import pandas as pd
import matplotlib.pyplot as plt

# Assuming your CSV filenames follow a pattern like "data2000.csv", "data2001.csv", ..., "data2022.csv"
# Adjust the path pattern according to your file naming and location
file_names = [f"Results_complexity/{year}.csv" for year in range(2000, 2023)]

# Initialize an empty list to store the sum of 'CR' column values from each file
data_sums = []

# Loop through the list of filenames, read each CSV file, and calculate the sum of the 'CR' column
for file_name in file_names:
    # Read the CSV file
    df = pd.read_csv(file_name)

    # Calculate the sum of the 'CR' column and append it to the list
    cr_sum = df['CR'].sum()  # Ensure you're using the correct column name
    data_sums.append(cr_sum)

# Create a list of years
years = list(range(2000, 2023))

# Plotting
plt.figure(figsize=(7, 2.5))
plt.plot(years, data_sums)

# Set the y-axis to start from 0
plt.ylim(bottom=0)

# Add title and axis labels
plt.ylabel("Degree of Complexity (CR)", fontsize=9)

# Hide the top and right axis lines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Only show the left and bottom axis lines
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')

# Save the chart as "dynamics.jpg"
plt.savefig("dynamics.jpg")

# Uncomment to display the chart
# plt.show()
