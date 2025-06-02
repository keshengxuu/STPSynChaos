# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:17:28 2024

@author: admin
"""

# Generate ranges for jee and jei
jee_values = [round(j * 0.1, 2) for j in range(21)]  # Values from 0.00 to 2.00 in steps of 0.05
jei_values = [round(j * 0.1, 2) for j in range(21)]  # Values from 0.00 to 2.00 in steps of 0.1

# Read Nonchaos_average_values.txt
with open('average_chaos.txt', 'r') as nv_file:
    chaos_values = nv_file.readlines()
    chaos_values = [line.strip() for line in chaos_values]

# Check if number of lines in Nonchaos_average_values.txt matches required number
if len(chaos_values) != 441:
    print("Error: varance_nonchaos.txt does not contain 861 lines.")
else:
    # Prepare the content to write into the txt file
    content = []
    index = 0
    for jee in jee_values:
        for jei in jei_values:
            # Append jee, jei, and corresponding data from Nonchaos_average_values.txt
            content.append(f'{jee:.2f}\t\t{jei:.2f}\t\t{chaos_values[index]}\n')
            index += 1

    # Write content to txt file
    with open('Chaos_avg.txt', 'w') as f:
        f.write('JEE\t\tJEI\t\tChaos_avg\n')
        f.writelines(content)

   